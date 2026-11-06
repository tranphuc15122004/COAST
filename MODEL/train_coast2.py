"""Standalone AC trainer for COAST2; legacy training entry points are untouched."""
from __future__ import annotations

import math
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

from baselines._critic import CriticBaseline
from layers import reinforce_loss
from MODEL.coast2_checkpoint import save_coast2_checkpoint
from MODEL.model.coast2 import (
    COAST2,
    continuation_loss,
    continuation_targets,
    materialize_coast2_model,
)
from problems import DVRPTW_Environment


def move_critic_to_device(critic: CriticBaseline, device):
    """Move the critic projection; Baseline.to() is a legacy no-op wrapper."""
    critic.project.to(device)


def make_optimizer(model: COAST2, critic: CriticBaseline, learner_lr=1e-4,
                   critic_lr=1e-3, weight_decay=0.0):
    lookahead_params = [p for p in model.lookahead_parameters() if p.requires_grad]
    lookahead_ids = {id(p) for p in lookahead_params}
    actor_params = [p for p in model.parameters() if p.requires_grad and id(p) not in lookahead_ids]
    critic_params = [p for p in critic.parameters() if p.requires_grad]
    groups = {"ac": actor_params + critic_params, "lookahead": lookahead_params}

    # Critic is part of the AC clipping group, while remaining a distinct LR group.
    all_optimizer_params = actor_params + lookahead_params + critic_params
    ids = [id(p) for p in all_optimizer_params]
    if len(ids) != len(set(ids)):
        raise ValueError("COAST2 optimizer parameter groups must be disjoint")
    if not actor_params or not lookahead_params or not critic_params:
        raise ValueError("COAST2 requires non-empty actor, lookahead, and critic parameter groups")
    optimizer = Adam([
        {"params": actor_params, "lr": float(learner_lr), "weight_decay": float(weight_decay)},
        {"params": lookahead_params, "lr": float(learner_lr), "weight_decay": float(weight_decay)},
        {"params": critic_params, "lr": float(critic_lr), "weight_decay": float(weight_decay)},
    ])
    return optimizer, groups


def clip_gradient_groups(ac_params, lookahead_params, max_norm=2.0):
    """Clip AC and predictor separately so auxiliary gradients cannot scale AC updates."""
    if max_norm is None:
        return 0.0, 0.0
    ac_norm = clip_grad_norm_(ac_params, max_norm) if ac_params else torch.tensor(0.0)
    lookahead_norm = (
        clip_grad_norm_(lookahead_params, max_norm)
        if lookahead_params else torch.tensor(0.0)
    )
    return float(ac_norm.detach().float().cpu().item()), float(
        lookahead_norm.detach().float().cpu().item()
    )


def _environment(data, nodes, cust_mask, args):
    return DVRPTW_Environment(
        data, nodes, cust_mask, args.pending_cost, args.late_cost
    )


def _rollout_training_batch(model, critic, data, nodes, cust_mask, args):
    env = _environment(data, nodes, cust_mask, args)
    env.reset()
    model._reset_customer_encoding()
    model._reset_memory(env)
    model.veh_speed = env.veh_speed

    logps, rewards, baseline_values, alive_steps, selected_contexts = [], [], [], [], []
    while not env.done:
        if env.new_customers:
            model._encode_customers(env.nodes, env.cust_mask)
        alive_steps.append((~env._finished).unsqueeze(-1).clone())
        logp, cache = model.score_step(env)
        action, chosen_logp, selected_context = model.select_action(env, logp, cache, greedy=False)
        baseline_values.append(critic.eval_step(env, cache["scores"], action))
        selected_contexts.append(selected_context)
        logps.append(chosen_logp)
        rewards.append(env.step(action))
    return logps, rewards, baseline_values, alive_steps, selected_contexts


def _pack_selected_context(context_steps):
    if not context_steps:
        raise ValueError("COAST2 rollout did not contain any transitions")
    return {
        key: torch.cat([context[key] for context in context_steps], dim=0)
        for key in context_steps[0]
    }


def _autocast(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def _saved_activation_context(device, enabled=False):
    """Offload tensors retained for backward to CPU, preserving the autograd graph."""
    if not enabled or torch.device(device).type != "cuda":
        return nullcontext()
    save_on_cpu = getattr(torch.autograd.graph, "save_on_cpu", None)
    if save_on_cpu is None:
        raise RuntimeError("This PyTorch version does not support saved-activation CPU offload")
    return save_on_cpu(pin_memory=False)


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_update(model, critic, optimizer, groups, data, nodes, cust_mask, args,
                 device, scaler=None):
    """Run one on-policy AC update plus selected-action continuation regression."""
    model.train()
    critic.project.train()
    if scaler is None:
        scaler = _make_scaler(bool(getattr(args, "amp", False)) and device.type == "cuda")
    amp_enabled = bool(getattr(args, "amp", False)) and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    def compute_update_loss():
        logps, rewards, baseline_values, alive_steps, selected_contexts = _rollout_training_batch(
            model, critic, data, nodes, cust_mask, args
        )
        ac_loss, ac_components = reinforce_loss(
            logps,
            rewards,
            baseline_values,
            adv_norm=args.adv_norm,
            entropy_coef=args.entropy_coef,
            return_components=True,
        )
        targets, valid = continuation_targets(rewards, alive_steps, discount=1.0)
        packed = _pack_selected_context(selected_contexts)
        prediction = model.postdecision_lookahead(packed).reshape_as(targets)
        aux_loss = continuation_loss(
            prediction, targets, valid,
            cost_scale=model.cost_scale,
            beta=float(getattr(args, "lookahead_huber_beta", 1.0)),
        )
        loss = ac_loss + float(getattr(args, "lookahead_loss_coef", 1.0)) * aux_loss
        mean_return = float(torch.stack(rewards).sum(0).mean().detach().float().item())
        metrics = {
            "loss": float(loss.detach().float().item()),
            "ac_loss": float(ac_loss.detach().float().item()),
            "continuation_loss": float(aux_loss.detach().float().item()),
            "policy_loss": float(ac_components["policy_loss"].detach().float().item()),
            "critic_loss": float(ac_components["critic_loss"].detach().float().item()),
            "entropy_loss": float(ac_components["entropy_loss"].detach().float().item()),
            "return": mean_return,
            "train_cost": -mean_return,
            "continuation_target_mean": float(targets[valid].mean().item()) if valid.any() else 0.0,
            "continuation_prediction_mean": float(prediction[valid].detach().mean().item()) if valid.any() else 0.0,
            "valid_transitions": int(valid.sum().item()),
            "gate": float(model.lookahead_gate.detach().item()),
        }
        return loss, metrics

    def finite_update(loss, metrics):
        keys = (
            "loss", "ac_loss", "continuation_loss", "policy_loss", "critic_loss",
            "entropy_loss", "return", "train_cost", "continuation_target_mean",
            "continuation_prediction_mean", "gate",
        )
        return math.isfinite(float(loss.detach().float().item())) and all(
            math.isfinite(float(metrics[key])) for key in keys
        )

    def compute_with_context(amp):
        with _saved_activation_context(
            device, enabled=bool(getattr(args, "activation_offload_cpu", False))
        ):
            with _autocast(device, amp):
                return compute_update_loss()

    loss, metrics = compute_with_context(amp_enabled)
    amp_fallback = False
    used_amp = amp_enabled
    if not finite_update(loss, metrics):
        if used_amp:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_with_context(amp=False)
            used_amp = False
            amp_fallback = True
        if not finite_update(loss, metrics):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("COAST2 produced a non-finite loss in AMP and FP32")

    if used_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    ac_params = groups["ac"]
    lookahead_params = groups["lookahead"]
    ac_norm, lookahead_norm = clip_gradient_groups(
        ac_params, lookahead_params, getattr(args, "max_grad_norm", 2.0)
    )
    metrics["ac_grad_norm"] = ac_norm
    metrics["lookahead_grad_norm"] = lookahead_norm
    if not math.isfinite(ac_norm) or not math.isfinite(lookahead_norm):
        optimizer.zero_grad(set_to_none=True)
        if used_amp:
            scaler.update()
            amp_fallback = True
            loss, metrics = compute_with_context(amp=False)
            if not finite_update(loss, metrics):
                optimizer.zero_grad(set_to_none=True)
                metrics.update({"amp_fallback": True, "successful": False,
                                "skip_reason": "nonfinite-fp32-fallback-loss"})
                return metrics
            loss.backward()
            ac_norm, lookahead_norm = clip_gradient_groups(
                groups["ac"], groups["lookahead"], getattr(args, "max_grad_norm", 2.0)
            )
            metrics["ac_grad_norm"] = ac_norm
            metrics["lookahead_grad_norm"] = lookahead_norm
        else:
            metrics.update({"amp_fallback": False, "successful": False,
                            "skip_reason": "nonfinite-gradient-norm"})
            return metrics
        if not math.isfinite(ac_norm) or not math.isfinite(lookahead_norm):
            optimizer.zero_grad(set_to_none=True)
            metrics.update({"amp_fallback": True, "successful": False,
                            "skip_reason": "nonfinite-fp32-fallback-gradient"})
            return metrics
        optimizer.step()
    elif used_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    metrics.update({"amp_fallback": amp_fallback, "successful": True, "skip_reason": None})
    return metrics


@torch.no_grad()
def evaluate_costs(model: COAST2, data, device="cpu", batch_size=512,
                   pending_cost=2.0, late_cost=1.0):
    if len(data) <= 0:
        raise ValueError("Evaluation dataset is empty")
    device = torch.device(device)
    loader = DataLoader(data, batch_size=int(batch_size), shuffle=False, num_workers=0)
    previous_training = model.training
    previous_greedy = model.greedy
    model.eval()
    model.greedy = True
    costs = []
    try:
        for batch in loader:
            if data.cust_mask is None:
                nodes, mask = batch.to(device), None
            else:
                nodes, mask = batch[0].to(device), batch[1].to(device)
            env = DVRPTW_Environment(data, nodes, mask, pending_cost, late_cost)
            _, _, rewards = model(env)
            costs.append(-torch.stack(rewards).sum(0).squeeze(-1).detach().float())
    finally:
        model.greedy = previous_greedy
        model.train(previous_training)
    result = torch.cat(costs)
    if result.numel() != len(data) or not torch.isfinite(result).all():
        raise FloatingPointError("COAST2 evaluation returned missing or non-finite costs")
    return result


def capture_coast2_training_state(scaler, completed_epoch, global_step):
    return {
        "epoch_complete": True,
        "completed_epoch": int(completed_epoch),
        "global_step": int(global_step),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "scaler": scaler.state_dict(),
    }


def restore_coast2_training_state(state, scaler):
    if not state or state.get("epoch_complete") is not True:
        raise ValueError("COAST2 supports exact resume only from completed epoch checkpoints")
    required = ("completed_epoch", "global_step", "python_rng", "numpy_rng", "torch_cpu_rng", "scaler")
    missing = [key for key in required if key not in state or state[key] is None]
    if missing:
        raise ValueError("COAST2 resume state lacks: {}".format(", ".join(missing)))
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    cuda_rng = state.get("torch_cuda_rng")
    if cuda_rng is not None:
        if not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count():
            raise RuntimeError("CUDA devices do not match the COAST2 resume checkpoint")
        torch.cuda.set_rng_state_all(cuda_rng)
    scaler.load_state_dict(state["scaler"])
    return int(state["completed_epoch"]), int(state["global_step"])
