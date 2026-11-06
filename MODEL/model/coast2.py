"""Isolated COAST2 model with post-decision continuation-cost guidance."""
from __future__ import annotations

import math
import inspect
import time
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .vectra import VECTRA


CONTEXT_FIELDS = (
    "vehicle_repr", "candidate_repr", "edge_repr", "remaining_repr",
    "active_state", "peer_state", "active_memory", "peer_memory", "scalars",
)


class PostDecisionLookahead(nn.Module):
    """Small candidate continuation-cost predictor with shared source projections."""

    def __init__(self, model_size: int, memory_size: int, rank: int = 32,
                 memory_rank: int = 16, hidden_size: int = 64):
        super().__init__()
        self.vehicle_projection = nn.Linear(model_size, rank)
        self.customer_projection = nn.Linear(model_size, rank)
        self.edge_projection = nn.Linear(model_size, rank)
        self.state_projection = nn.Linear(5, rank)
        self.memory_projection = nn.Linear(memory_size, memory_rank)
        input_size = 6 * rank + 2 * memory_rank + 8
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.input_size = input_size

    def forward(self, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_count = context["candidate_repr"].size(1)

        def candidate_expand(value):
            return value.expand(-1, candidate_count, -1) if value.size(1) == 1 else value

        projected = tuple(candidate_expand(value) for value in (
            self.vehicle_projection(context["vehicle_repr"]),
            self.customer_projection(context["candidate_repr"]),
            self.edge_projection(context["edge_repr"]),
            self.customer_projection(context["remaining_repr"]),
            self.state_projection(context["active_state"]),
            self.state_projection(context["peer_state"]),
            self.memory_projection(context["active_memory"]),
            self.memory_projection(context["peer_memory"]),
            context["scalars"],
        ))
        features = torch.cat(projected, dim=-1)
        return self.predictor(features).squeeze(-1)


class COAST2(VECTRA):
    """COAST variant with separate lookahead; inputs use dataset-normalized time units."""

    def __init__(self, cust_feat_size: int, veh_state_size: int, *,
                 cost_scale: float = 10.0, scalar_count_scale: float = 50.0,
                 horizon_scale: float = 1.0, lookahead_rank: int = 32,
                 lookahead_memory_rank: int = 16, lookahead_hidden_size: int = 64,
                 gate_init: float = 0.05, activation_checkpointing: bool = False, **kwargs):
        for flag in ("use_edge_features", "use_memory", "use_ownership"):
            if kwargs.get(flag, True) is False:
                raise ValueError("COAST2 requires {}=True for the registered C+O+L method".format(flag))
            kwargs[flag] = True
        kwargs["use_lookahead"] = False
        super().__init__(cust_feat_size, veh_state_size, **kwargs)
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")
        if cost_scale <= 0 or scalar_count_scale <= 0 or horizon_scale <= 0:
            raise ValueError("COAST2 scales must be positive")

        # The old head is deliberately absent from deployed COAST2 state/parameters.
        del self.lookahead_head
        memory_size = self.coord_memory.hidden_size
        # Keep the legacy COAST initialization stream unchanged for the AC/critic.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(torch.initial_seed() + 0xC0A57)
            self.postdecision_lookahead = PostDecisionLookahead(
                self.model_size, memory_size, lookahead_rank,
                lookahead_memory_rank, lookahead_hidden_size,
            )
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))
        self.cost_scale = float(cost_scale)
        self.scalar_count_scale = float(scalar_count_scale)
        # Dataset.normalize() already divides times by the physical horizon.
        # Historical checkpoints explicitly retain their former scale of 480.
        self.horizon_scale = float(horizon_scale)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.selective_customer_encoding and self.activation_checkpointing:
            raise ValueError("Cannot combine activation_checkpointing with selective_customer_encoding")
        resolved = inspect.signature(VECTRA.__init__).bind_partial(
            None, cust_feat_size, veh_state_size, **kwargs
        )
        resolved.apply_defaults()
        resolved_config = {key: value for key, value in resolved.arguments.items() if key != "self"}
        self.coast2_config = {
            **resolved_config,
            "cost_scale": self.cost_scale,
            "scalar_count_scale": self.scalar_count_scale,
            "horizon_scale": self.horizon_scale,
            "lookahead_rank": int(lookahead_rank),
            "lookahead_memory_rank": int(lookahead_memory_rank),
            "lookahead_hidden_size": int(lookahead_hidden_size),
            "gate_init": float(gate_init),
        }
        self.coast2_config["memory_size"] = int(memory_size)

    @property
    def lookahead_gate(self) -> torch.Tensor:
        return self.gate_logit.sigmoid()

    def lookahead_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.postdecision_lookahead.parameters()

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        lookahead_ids = {id(p) for p in self.lookahead_parameters()}
        return (p for p in self.parameters() if id(p) not in lookahead_ids)

    def _checkpoint_training_block(self, function, *inputs):
        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(function, *inputs, use_reentrant=False)
        return function(*inputs)

    def _encode_customers(self, customers, mask=None):
        if self.selective_customer_encoding and self.activation_checkpointing:
            raise ValueError("Cannot combine activation_checkpointing with selective_customer_encoding")

        if self.selective_customer_encoding:
            return super()._encode_customers(customers, mask)

        if not (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return super()._encode_customers(customers, mask)

        start = time.perf_counter()
        customer_input = customers.clone()
        mask_input = None if mask is None else mask.clone()

        def encode_customer_state(nodes):
            customer_embedding = torch.cat((
                self.depot_embedding(nodes[:, 0:1, :]),
                self.cust_embedding(nodes[:, 1:, :]),
            ), dim=1)
            if mask_input is not None:
                customer_embedding[mask_input] = 0
            use_bottleneck = (
                self.latent_bottleneck
                and self.latent_tokens is not None
                and self.latent_tokens > 1
                and self.latent_tokens < nodes.size(1)
                and nodes.size(1) >= self.latent_min_nodes
            )
            if use_bottleneck:
                customer_encoding = self._encode_customers_bottleneck(
                    customer_embedding, nodes, mask_input
                )
            else:
                customer_encoding = self.cust_encoder(
                    customer_embedding, mask_input, coords=nodes[:, :, :2]
                )
            customer_repr = self.dropout(self.cust_project(customer_encoding))
            if mask_input is not None:
                customer_repr[mask_input] = 0
            return customer_encoding, customer_repr

        self.cust_enc, self.cust_repr = self._checkpoint_training_block(
            encode_customer_state, customer_input
        )
        self._forward_profiler.add("encode_customers", time.perf_counter() - start)

    def _postdecision_context(self, dyna, veh_repr, edge_repr) -> Dict[str, torch.Tensor]:
        """Build raw B×candidate contexts without stepping or mutating the environment."""
        if dyna.nodes.size(-1) < 7:
            raise ValueError("COAST2 requires the seven-feature DVRPTW node representation")

        batch_size, node_count, _ = dyna.nodes.shape
        device, dtype = dyna.nodes.device, dyna.nodes.dtype
        candidate_repr = self.cust_repr
        candidate_edge = edge_repr.squeeze(1)
        active_idx = dyna.cur_veh_idx.reshape(batch_size, 1)
        active_state_raw = dyna.vehicles.gather(
            1, active_idx[:, :, None].expand(-1, -1, dyna.vehicles.size(-1))
        )
        active_done = dyna.veh_done.gather(1, active_idx).to(dtype).unsqueeze(-1)
        active_state_raw = torch.cat((active_state_raw, active_done), dim=-1)

        vehicle_ids = torch.arange(dyna.veh_count, device=device).expand(batch_size, -1)
        peer_mask = vehicle_ids.ne(active_idx)
        peer_states = torch.cat((dyna.vehicles, dyna.veh_done.to(dtype).unsqueeze(-1)), dim=-1)
        peer_count = peer_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
        peer_state = (
            peer_states * peer_mask.unsqueeze(-1).to(dtype)
        ).sum(dim=1, keepdim=True) / peer_count.unsqueeze(-1)
        if dyna.veh_count == 1:
            peer_state = torch.zeros_like(peer_state)
        peer_state = peer_state.clone()
        peer_state[:, :, 3] = peer_state[:, :, 3] / self.horizon_scale

        if self._veh_memory is None:
            memories = dyna.vehicles.new_zeros(
                (batch_size, dyna.veh_count, self.coord_memory.hidden_size)
            )
        else:
            memories = self._veh_memory
        active_memory = memories.gather(
            1, active_idx[:, :, None].expand(-1, -1, memories.size(-1))
        )
        memory_mask = peer_mask.to(dtype).unsqueeze(-1)
        peer_memory = (memories * memory_mask).sum(dim=1, keepdim=True) / peer_count.unsqueeze(-1)
        if dyna.veh_count == 1:
            peer_memory = torch.zeros_like(peer_memory)

        padding = dyna._padding_mask()
        known = ~dyna.cust_mask & ~dyna.served & ~padding
        if node_count:
            known = known.clone()
            known[:, 0] = False
        known_f = known.to(dtype)
        known_count = known_f.sum(dim=1, keepdim=True)
        candidate_known = known
        remain_count = known_count - candidate_known.to(dtype)
        remain_denom = remain_count.clamp_min(1.0)

        pool_sum = (candidate_repr * known_f.unsqueeze(-1)).sum(dim=1, keepdim=True)
        pool_after = pool_sum - candidate_repr * candidate_known.unsqueeze(-1).to(dtype)
        remaining_repr = pool_after / remain_denom.unsqueeze(-1)
        remaining_repr = remaining_repr.masked_fill(remain_count.unsqueeze(-1) <= 0, 0.0)

        nodes = dyna.nodes
        feature_mask = known_f.unsqueeze(-1)
        demand_sum = (nodes[:, :, 2:3] * feature_mask).sum(dim=1, keepdim=True)
        field_sums = torch.cat((
            demand_sum,
            (nodes[:, :, 3:6] * feature_mask).sum(dim=1, keepdim=True),
            (nodes[:, :, :2] * feature_mask).sum(dim=1, keepdim=True),
        ), dim=-1)
        candidate_stats = torch.cat((nodes[:, :, 2:3], nodes[:, :, 3:6], nodes[:, :, :2]), dim=-1)
        stats_after = (
            field_sums.expand(-1, node_count, -1)
            - candidate_stats * candidate_known.unsqueeze(-1).to(dtype)
        )
        count_after = remain_count.expand(-1, node_count)
        means_after = stats_after[:, :, 1:] / count_after.clamp_min(1.0).unsqueeze(-1)
        means_after = means_after.masked_fill(count_after.unsqueeze(-1) <= 0, 0.0)
        capacity_scale = max(float(dyna.veh_capa), 1e-8)
        current_time = active_state_raw[:, :, 3:4].expand(-1, node_count, -1)
        scalars = torch.cat((
            (count_after / self.scalar_count_scale).unsqueeze(-1),
            (stats_after[:, :, :1] / capacity_scale),
            means_after[:, :, 0:1] / self.horizon_scale,
            means_after[:, :, 1:2] / self.horizon_scale,
            means_after[:, :, 2:3] / self.horizon_scale,
            means_after[:, :, 3:4],
            means_after[:, :, 4:5],
            current_time / self.horizon_scale,
        ), dim=-1)

        active_position = active_state_raw[:, :, :2]
        candidate_position = nodes[:, :, :2]
        distance = torch.pairwise_distance(
            active_position.expand(-1, node_count, -1), candidate_position
        )
        travel_time = distance / float(getattr(dyna, "veh_speed", 1.0))
        arrival = torch.maximum(
            active_state_raw[:, :, 3].expand(-1, node_count) + travel_time,
            nodes[:, :, 3],
        )
        post_available = (arrival + nodes[:, :, 5]) / self.horizon_scale
        post_capacity = active_state_raw[:, :, 2].expand(-1, node_count) - nodes[:, :, 2]
        is_depot = torch.zeros((batch_size, node_count), device=device, dtype=dtype)
        is_depot[:, 0] = 1.0
        active_post_state = torch.cat((
            candidate_position,
            post_capacity.unsqueeze(-1),
            post_available.unsqueeze(-1),
            is_depot.unsqueeze(-1),
        ), dim=-1)

        context = {
            "vehicle_repr": veh_repr,
            "candidate_repr": candidate_repr,
            "edge_repr": candidate_edge,
            "remaining_repr": remaining_repr,
            "active_state": active_post_state,
            "peer_state": peer_state,
            "active_memory": active_memory,
            "peer_memory": peer_memory,
            "scalars": scalars,
        }
        return {name: value.detach() for name, value in context.items()}

    def _base_scores_pre_tanh(self, veh_repr, edge_emb, owner_bias, mask,
                              cust_repr=None):
        if cust_repr is None:
            cust_repr = self.cust_repr
        att_score = self.cross_fusion(
            veh_repr, cust_repr, self._compatibility_edge_embedding(edge_emb)
        )

        score_mask = mask.clone()
        all_masked = score_mask.all(dim=2, keepdim=True)
        if all_masked.any():
            rows = all_masked.squeeze(2).squeeze(1)
            score_mask[rows, 0, 0] = False

        def valid_z_norm(source):
            valid = (~score_mask).to(source.dtype)
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            centered = (source - (source * valid).sum(dim=-1, keepdim=True) / count) * valid
            variance = centered.square().sum(dim=-1, keepdim=True) / count
            return (centered / variance.clamp_min(1e-8).sqrt()).masked_fill(score_mask, 0.0)

        zeros = torch.zeros_like(att_score)
        combined = torch.stack((valid_z_norm(att_score), valid_z_norm(owner_bias), zeros), dim=-1)
        if self.fusion_mode == "mlp":
            scores = self.score_fusion(combined).squeeze(-1)
        else:
            scores = (combined * self.linear_fusion_weights.to(combined.dtype)).sum(dim=-1)
        return scores.masked_fill(score_mask, 0.0), score_mask

    def score_step(self, dyna) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Score legal actions and expose a detached raw context for selected-action loss."""
        checkpoint_active = (
            self.activation_checkpointing and self.training and torch.is_grad_enabled()
        )
        cust_repr = self.cust_repr
        if checkpoint_active:
            # Environment state mutates after this step, so checkpoint inputs need snapshots.
            vehicles = dyna.vehicles.detach().clone()
            veh_idx = dyna.cur_veh_idx.detach().clone()
            veh_mask = dyna.cur_veh_mask.detach().clone()

            def encode_active_vehicle(vehicle_state, customers, active_idx, active_mask):
                state = vehicle_state.gather(
                    1, active_idx[:, :, None].expand(-1, -1, vehicle_state.size(-1))
                )
                if active_mask.dim() == 3 and active_mask.size(1) != 1:
                    active_mask = active_mask.gather(
                        1, active_idx[:, :, None].expand(-1, -1, active_mask.size(-1))
                    )
                return self._encode_fleet(state, customers, mask=active_mask)

            veh_repr = self._checkpoint_training_block(
                encode_active_vehicle, vehicles, cust_repr, veh_idx, veh_mask
            )
            edge_features = self._build_edge_features(
                vehicles, dyna.nodes, veh_idx, veh_mask
            )
            edge_emb = self._checkpoint_training_block(self.edge_encoder, edge_features)

            def owner_bias_for_state(memory, customers, active_idx):
                logits = self.owner_head(memory, customers)
                owner_prob = logits.softmax(dim=1)
                owner_bias = owner_prob.gather(
                    1, active_idx[:, :, None].expand(-1, -1, owner_prob.size(-1))
                )
                return owner_bias.clamp_min(1e-9).log()

            if self._veh_memory is None:
                owner_bias = cust_repr.new_zeros(
                    (cust_repr.size(0), 1, cust_repr.size(1))
                )
            else:
                owner_bias = self._checkpoint_training_block(
                    owner_bias_for_state, self._veh_memory, cust_repr, veh_idx
                )
            score_mask_input = veh_mask
        else:
            veh_idx = dyna.cur_veh_idx
            score_mask_input = dyna.cur_veh_mask
            veh_repr = self._repr_vehicle(dyna.vehicles, veh_idx, score_mask_input)
            edge_emb = self._compute_edge_embedding(
                dyna.vehicles, dyna.nodes, veh_idx, score_mask_input
            )
            owner_bias = self._compute_owner_bias(veh_idx)
        device_type = veh_repr.device.type
        try:
            autocast_enabled = torch.is_autocast_enabled(device_type)
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        # Do not cache detached low-precision head weights here: the selected-action
        # regression calls this same head later in the outer autocast context.
        with torch.autocast(
            device_type=device_type, enabled=autocast_enabled, cache_enabled=False
        ):
            with torch.no_grad():
                context = self._postdecision_context(dyna, veh_repr, edge_emb)
                predicted_cost_scaled = self.postdecision_lookahead(context)

        base_scores, score_mask = self._checkpoint_training_block(
            self._base_scores_pre_tanh,
            veh_repr, edge_emb, owner_bias, score_mask_input, cust_repr,
        )
        valid = (~score_mask).to(predicted_cost_scaled.dtype)
        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_prediction = (predicted_cost_scaled.unsqueeze(1) * valid).sum(
            dim=-1, keepdim=True
        ) / valid_count
        centered_cost = predicted_cost_scaled.unsqueeze(1) - mean_prediction
        scores = base_scores - self.lookahead_gate * self.cost_scale * centered_cost
        scores = scores.masked_fill(score_mask, 0.0)
        if self.tanh_xplor is not None:
            scores = self.tanh_xplor * scores.tanh()
        logp = self._get_logp(scores, score_mask)
        cache = {
            "context": context,
            "veh_repr": veh_repr,
            "edge_emb": edge_emb,
            "scores": scores,
            "predicted_cost_scaled": predicted_cost_scaled.unsqueeze(1),
        }
        return logp, cache

    def select_action(self, dyna, logp, cache, greedy=None):
        use_greedy = self.greedy if greedy is None else bool(greedy)
        action = logp.argmax(dim=1, keepdim=True) if use_greedy else logp.exp().multinomial(1)
        selected_context = select_candidate_context(cache["context"], action)
        self._update_memory(dyna.cur_veh_idx, action, cache["veh_repr"], cache["edge_emb"])
        return action, logp.gather(1, action), selected_context

    def step(self, dyna):
        logp, cache = self.score_step(dyna)
        action, chosen_logp, _ = self.select_action(dyna, logp, cache)
        return action, chosen_logp


def select_candidate_context(context: Dict[str, torch.Tensor], candidate_idx: torch.Tensor):
    """Gather and own detached selected-action inputs for later regression."""
    batch_size = candidate_idx.size(0)
    selected = {}
    for name in CONTEXT_FIELDS:
        value = context[name].detach()
        if value.dim() >= 3 and value.size(0) == batch_size and value.size(1) > 1:
            index = candidate_idx.long()
            index = index.view(batch_size, 1, *([1] * (value.dim() - 2)))
            index = index.expand(batch_size, 1, *value.shape[2:])
            value = value.gather(1, index)
        selected[name] = value.clone()
    return selected


def continuation_targets(rewards, alive_before_step, discount: float = 1.0):
    """Return detached negative MC continuation costs, excluding each step's reward."""
    if len(rewards) != len(alive_before_step):
        raise ValueError("rewards and alive_before_step must have equal lengths")
    if not rewards:
        raise ValueError("at least one transition is required")
    future = torch.zeros_like(rewards[0], dtype=torch.float32)
    targets = [None] * len(rewards)
    valid = []
    for step in range(len(rewards) - 1, -1, -1):
        targets[step] = -future
        reward = rewards[step].detach().to(torch.float32)
        alive = alive_before_step[step].detach().to(device=reward.device, dtype=torch.bool)
        reward = reward.masked_fill(~alive, 0.0)
        future = reward + float(discount) * future
        valid.append(alive)
    valid.reverse()
    return torch.stack(targets).detach(), torch.stack(valid).detach()


def continuation_loss(prediction, target, valid, cost_scale: float = 10.0, beta: float = 1.0):
    """Smooth-L1 on valid selected actions; empty masks produce a differentiable zero."""
    if cost_scale <= 0:
        raise ValueError("cost_scale must be positive")
    target = target.detach().to(device=prediction.device, dtype=torch.float32) / cost_scale
    valid = valid.detach().to(device=prediction.device, dtype=torch.bool)
    valid = torch.broadcast_to(valid, prediction.shape)
    if not valid.any():
        return prediction.sum() * 0.0
    errors = F.smooth_l1_loss(
        prediction.to(torch.float32), target, reduction="none", beta=beta
    )
    return errors.masked_select(valid).mean()


def materialize_coast2_model(model: COAST2, *, preserve_rng: bool = True):
    """Initialize inherited LazyLinear layers before optimizer/checkpoint creation."""
    cpu_rng = torch.get_rng_state() if preserve_rng else None
    cuda_rng = torch.cuda.get_rng_state_all() if preserve_rng and torch.cuda.is_initialized() else None
    try:
        lazy_inputs = (
            (model.fleet_encoder.input_proj, (1, model.coord_memory.veh_state_size)),
            (model.owner_head.veh_proj, (1, 1, model.coord_memory.hidden_size)),
            (model.coord_memory.input_proj, (1, 3 * model.model_size)),
        )
        for module, shape in lazy_inputs:
            if hasattr(module, "initialize_parameters"):
                module.initialize_parameters(
                    torch.empty(*shape, device=model.gate_logit.device)
                )
    finally:
        if cpu_rng is not None:
            torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return model
