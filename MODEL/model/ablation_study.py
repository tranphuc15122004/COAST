"""Models and contracts for the locked two-branch COAST ablation study."""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from .vectra import COAST, VECTRA


ROLE_AWARE_GENERIC_OWNERSHIP = "role_aware_generic_memory"


@dataclass(frozen=True)
class StudyVariantSpec:
    name: str
    label: str
    use_edge_features: bool
    use_memory: bool
    use_ownership: bool
    compatibility_uses_edges: bool
    use_lookahead: bool
    ownership_scorer: str


_VARIANTS = {
    "f11": StudyVariantSpec("f11", "C + O", True, True, True, True, False, "ownership"),
    "f10": StudyVariantSpec("f10", "C", True, False, False, True, False, "none"),
    "f01": StudyVariantSpec("f01", "C⁻ᴱ + O", True, True, True, False, False, "ownership"),
    "cm": StudyVariantSpec(
        "cm", "C + G", True, True, True, True, False, ROLE_AWARE_GENERIC_OWNERSHIP
    ),
}


def variant_spec(name: str) -> StudyVariantSpec:
    try:
        return _VARIANTS[name.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            "Unknown COAST ablation variant {!r}; choose one of {}".format(
                name, ", ".join(_VARIANTS)
            )
        ) from None


class RoleAwareMemoryScoringHead(nn.Module):
    """Capacity-matched generic score from customer, active memory, and peer memory."""

    def __init__(self, model_size: int, memory_size: int):
        super().__init__()
        self.model_size = int(model_size)
        self.memory_size = int(memory_size)
        target_owner_params = memory_size * model_size + model_size + model_size * model_size
        projection_params = memory_size * model_size
        hidden_size = max(
            1,
            (target_owner_params - projection_params - 1) // (3 * model_size + 2),
        )
        self.hidden_size = int(hidden_size)
        self.memory_projection = nn.Linear(memory_size, model_size, bias=False)
        self.context_projection = nn.Linear(3 * model_size, self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, 1)

    def forward(
        self,
        veh_memory: torch.Tensor,
        cust_repr: torch.Tensor,
        active_vehicle_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate scores of shape ``N × 1 × L_c``.

        Fleet memories are separated structurally into the active vehicle and
        the mean of its peers. No raw peer vehicle state or edge feature enters
        this generic Ownership-slot control.
        """
        if veh_memory.dim() != 3 or cust_repr.dim() != 3:
            raise ValueError("veh_memory and cust_repr must be rank-3 tensors")
        batch_size, vehicle_count, memory_dim = veh_memory.shape
        if memory_dim != self.memory_size:
            raise ValueError(
                "Expected memory dimension {}, got {}".format(self.memory_size, memory_dim)
            )
        if cust_repr.size(0) != batch_size or cust_repr.size(-1) != self.model_size:
            raise ValueError("cust_repr batch/model dimensions do not match the scorer")

        active_idx = active_vehicle_idx.reshape(batch_size, 1).long()
        if ((active_idx < 0) | (active_idx >= vehicle_count)).any():
            raise ValueError("active_vehicle_idx is outside the fleet memory")

        memory = self.memory_projection(veh_memory)
        active = memory.gather(1, active_idx[:, :, None].expand(-1, -1, self.model_size))
        peer_mask = torch.ones(
            (batch_size, vehicle_count), device=memory.device, dtype=memory.dtype
        )
        peer_mask.scatter_(1, active_idx, 0.0)
        peer_count = peer_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        peers = (memory * peer_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / peer_count.unsqueeze(-1)
        if vehicle_count == 1:
            peers = torch.zeros_like(peers)

        active = active.expand(-1, cust_repr.size(1), -1)
        peers = peers.expand(-1, cust_repr.size(1), -1)
        context = torch.cat((cust_repr, active, peers), dim=-1)
        hidden = torch.relu(self.context_projection(context))
        return self.output_projection(hidden).transpose(1, 2)


class _CompatibilityWithoutEdgesCOAST(COAST):
    """F01: keep raw edges for memory updates, mask only the Compatibility path."""

    def _compatibility_edge_embedding(self, edge_emb):
        return torch.zeros_like(edge_emb)


_CompatibilityWithoutEdgesVECTRA = _CompatibilityWithoutEdgesCOAST


class _GenericMemoryOwnershipCOAST(COAST):
    """CM: replace only the Ownership score head with a generic memory scorer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.owner_head
        self.generic_memory_scorer = RoleAwareMemoryScoringHead(
            self.model_size, self.coord_memory.hidden_size
        )

    def _compute_owner_bias(self, cur_veh_idx):
        if not self.use_ownership or self._veh_memory is None:
            return self.cust_repr.new_zeros(
                (self.cust_repr.size(0), 1, self.cust_repr.size(1))
            )
        owner_start = time.perf_counter()
        score = self.generic_memory_scorer(
            self._veh_memory, self.cust_repr, cur_veh_idx
        )
        self._forward_profiler.add(
            "ownership_head", time.perf_counter() - owner_start
        )
        return score


_GenericMemoryOwnershipVECTRA = _GenericMemoryOwnershipCOAST


def build_ablation_model(
    name: str,
    cust_feat_size: int,
    veh_state_size: int,
    selective_customer_encoding: bool = False,
    **kwargs,
) -> COAST:
    """Build a study model while enforcing that variant's edge/memory contract."""
    spec = variant_spec(name)
    required = {
        "use_edge_features": spec.use_edge_features,
        "use_memory": spec.use_memory,
        "use_ownership": spec.use_ownership,
        "use_lookahead": spec.use_lookahead,
    }
    for key, expected in required.items():
        if key in kwargs and kwargs[key] != expected:
            raise ValueError(
                "Variant {} requires {}={}, got {}".format(name, key, expected, kwargs[key])
            )
        kwargs[key] = expected

    model_class = COAST
    if name.lower() == "f01":
        model_class = _CompatibilityWithoutEdgesCOAST
    elif name.lower() == "cm":
        model_class = _GenericMemoryOwnershipCOAST
    model = model_class(
        cust_feat_size,
        veh_state_size,
        selective_customer_encoding=selective_customer_encoding,
        **kwargs,
    )
    if not spec.use_memory:
        del model.coord_memory
    if not spec.use_ownership:
        del model.owner_head
    if not spec.use_lookahead:
        del model.lookahead_head
    model.variant_name = spec.name
    model.study_label = spec.label
    model.compatibility_uses_edges = spec.compatibility_uses_edges
    model.ownership_scorer = spec.ownership_scorer
    return model
