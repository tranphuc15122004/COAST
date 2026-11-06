from MODEL.model import COAST, VECTRA
from problems import *
from baselines import *
from externals import *
from dep import *
from utils import *

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import time
import os
import math
from itertools import chain


def save_best_val_checkpoint(args, ep, learner, optim, baseline = None, lr_sched = None, best_val_mu = None):
	checkpoint = {
			"ep": ep,
			"best_ep": ep,
			"best_val_mu": None if best_val_mu is None else float(best_val_mu),
			"model": learner.state_dict(),
			"optim": optim.state_dict()
			}
	if args.rate_decay is not None:
		checkpoint["lr_sched"] = lr_sched.state_dict()
	if args.baseline_type == "critic":
		checkpoint["critic"] = baseline.state_dict()
	torch.save(checkpoint, os.path.join(args.output_dir, "chkpt_best.pyth"))

def _compute_step_outputs(learner, baseline, dyna):
	veh_repr = learner._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.mask)

	if hasattr(learner, "_compute_edge_embedding") and hasattr(learner, "_compute_owner_bias") and hasattr(learner, "_compute_lookahead"):
		edge_emb = learner._compute_edge_embedding(
			dyna.vehicles,
			dyna.nodes,
			dyna.cur_veh_idx,
			dyna.cur_veh_mask,
		)
		owner_bias = learner._compute_owner_bias(dyna.cur_veh_idx)
		lookahead = learner._compute_lookahead(veh_repr, learner.cust_repr, edge_emb)
		compat = learner._score_customers(
			veh_repr,
			learner.cust_repr,
			edge_emb,
			owner_bias,
			lookahead,
			dyna.cur_veh_mask,
		)
	else:
		edge_emb = None
		compat = learner._score_customers(veh_repr)

	logp = learner._get_logp(compat, dyna.cur_veh_mask)
	return compat, logp, veh_repr, edge_emb


def _collect_rollout(args, dataset, minibatch, Environment: VRP_Environment, env_params, baseline, device):
	if dataset.cust_mask is None:
		custs, mask = minibatch.to(device), None
	else:
		custs, mask = minibatch[0].to(device), minibatch[1].to(device)

	dyna = Environment(dataset, custs, mask, *env_params)
	actions, logps, rewards, bl_vals = baseline(dyna)

	old_logps = torch.stack(logps, dim = 0).detach()
	if isinstance(rewards, torch.Tensor):
		rewards_t = rewards.unsqueeze(0).detach()
	else:
		rewards_t = torch.stack(rewards, dim = 0).detach()

	if isinstance(bl_vals, torch.Tensor):
		values_t = bl_vals.unsqueeze(0).detach()
	elif bl_vals is None or len(bl_vals) == 0:
		values_t = torch.zeros_like(rewards_t)
	else:
		values_t = torch.stack(bl_vals, dim = 0).detach()

	if values_t.size(0) != rewards_t.size(0):
		values_t = values_t[:rewards_t.size(0)]

	action_idx = torch.stack([a[1] for a in actions], dim = 0).detach()

	gamma = args.ppo_gamma
	lam = args.ppo_gae_lambda
	advantages = torch.zeros_like(rewards_t)
	returns = torch.zeros_like(rewards_t)
	gae = torch.zeros_like(rewards_t[0])
	next_value = torch.zeros_like(values_t[0])

	for t in reversed(range(rewards_t.size(0))):
		delta = rewards_t[t] + gamma * next_value - values_t[t]
		gae = delta + gamma * lam * gae
		advantages[t] = gae
		returns[t] = gae + values_t[t]
		next_value = values_t[t]

	if args.ppo_adv_norm:
		adv_flat = advantages.view(-1)
		advantages = (advantages - adv_flat.mean()) / adv_flat.std(unbiased = False).clamp_min(1e-8)

	ep_return = rewards_t.sum(dim = 0).mean().item()
	mean_bl = values_t.mean().item()

	return {
		"dataset": dataset,
		"custs": custs,
		"mask": mask,
		"action_idx": action_idx,
		"old_logps": old_logps,
		"returns": returns.detach(),
		"advantages": advantages.detach(),
		"ep_return": ep_return,
		"mean_bl": mean_bl,
	}


def _evaluate_rollout(learner, baseline, Environment: VRP_Environment, env_params, rollout):
	dyna = Environment(rollout["dataset"], rollout["custs"], rollout["mask"], *env_params)
	dyna.reset()

	cust_mask = getattr(dyna, "cust_mask", None)
	learner._encode_customers(dyna.nodes, cust_mask)
	if hasattr(dyna, "veh_speed"):
		learner.veh_speed = dyna.veh_speed
	if hasattr(learner, "_reset_memory"):
		learner._reset_memory(dyna)

	logps_new, values_new, entropies = [], [], []
	t_steps = rollout["action_idx"].size(0)
	for t in range(t_steps):
		compat, logp, veh_repr, edge_emb = _compute_step_outputs(learner, baseline, dyna)
		chosen = rollout["action_idx"][t]

		step_logp = logp.gather(1, chosen)
		step_val = baseline.eval_step(dyna, compat, chosen)

		probs = logp.exp()
		logp_safe = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
		step_entropy = -(probs * logp_safe).sum(dim = 1, keepdim = True)

		if hasattr(learner, "_update_memory") and edge_emb is not None:
			learner._update_memory(dyna.cur_veh_idx, chosen, veh_repr, edge_emb)
		dyna.step(chosen)

		logps_new.append(step_logp)
		values_new.append(step_val)
		entropies.append(step_entropy)

	logps_new = torch.stack(logps_new, dim = 0)
	values_new = torch.stack(values_new, dim = 0)
	entropies = torch.stack(entropies, dim = 0)
	return logps_new, values_new, entropies


def _ppo_update(args, rollout, learner, baseline, Environment, env_params, optim, scaler):
	last_policy_loss = 0.0
	last_value_loss = 0.0
	last_entropy = 0.0
	last_kl = 0.0
	grad_norm = torch.tensor(0.0, device = rollout["old_logps"].device)

	for _ in range(args.ppo_epochs):
		with autocast(enabled = args.amp):
			new_logps, new_values, entropies = _evaluate_rollout(
				learner,
				baseline,
				Environment,
				env_params,
				rollout,
			)

			ratio = (new_logps - rollout["old_logps"]).exp()
			adv = rollout["advantages"]
			clipped_ratio = ratio.clamp(1.0 - args.ppo_clip_range, 1.0 + args.ppo_clip_range)
			policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

			value_loss = F.smooth_l1_loss(new_values, rollout["returns"])
			entropy_bonus = entropies.mean()

			loss = policy_loss + args.ppo_value_coef * value_loss - args.ppo_entropy_coef * entropy_bonus

		optim.zero_grad()
		if args.amp:
			scaler.scale(loss).backward()
			if args.max_grad_norm is not None:
				scaler.unscale_(optim)
				grad_norm = clip_grad_norm_(
					chain.from_iterable(grp["params"] for grp in optim.param_groups),
					args.max_grad_norm,
				)
			scaler.step(optim)
			scaler.update()
		else:
			loss.backward()
			if args.max_grad_norm is not None:
				grad_norm = clip_grad_norm_(
					chain.from_iterable(grp["params"] for grp in optim.param_groups),
					args.max_grad_norm,
				)
			optim.step()

		approx_kl = (rollout["old_logps"] - new_logps).mean().detach()

		last_policy_loss = policy_loss.detach().item()
		last_value_loss = value_loss.detach().item()
		last_entropy = entropy_bonus.detach().item()
		last_kl = approx_kl.item()

		if args.ppo_target_kl is not None and approx_kl > args.ppo_target_kl:
			break

	return {
		"loss": last_policy_loss + args.ppo_value_coef * last_value_loss - args.ppo_entropy_coef * last_entropy,
		"policy_loss": last_policy_loss,
		"value_loss": last_value_loss,
		"entropy": last_entropy,
		"kl": last_kl,
		"grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
	}


def train_epoch(args, data, Environment: VRP_Environment, env_params, baseline, optim, device, ep, scaler):
	baseline.learner.train()
	baseline.project.train()
	loader = DataLoader(
		data,
		args.batch_size,
		True,
		num_workers = args.num_workers,
		pin_memory = args.pin_memory,
		persistent_workers = args.num_workers > 0,
	)

	ep_loss = 0.0
	ep_prob = 0.0
	ep_val = 0.0
	ep_bl = 0.0
	ep_norm = 0.0
	with tqdm(loader, desc = "PPO Ep.#{: >3d}/{: <3d}".format(ep + 1, args.epoch_count)) as progress:
		for minibatch in progress:
			rollout = _collect_rollout(args, data, minibatch, Environment, env_params, baseline, device)
			upd = _ppo_update(args, rollout, baseline.learner, baseline, Environment, env_params, optim, scaler)

			prob = rollout["old_logps"].exp().mean().item()
			ep_loss += upd["loss"]
			ep_prob += prob
			ep_val += rollout["ep_return"]
			ep_bl += rollout["mean_bl"]
			ep_norm += upd["grad_norm"]

			progress.set_postfix_str(
				"l={:.4g} p={:9.4g} ret={:6.4g} v={:6.4g} kl={:6.4g} |g|={:.4g}".format(
					upd["loss"],
					prob,
					rollout["ep_return"],
					rollout["mean_bl"],
					upd["kl"],
					upd["grad_norm"],
				)
			)

	return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))


def test_epoch(args, test_env, learner, ref_costs):
	learner.eval()
	with torch.no_grad():
		if args.problem_type[0] == "s":
			costs = test_env.nodes.new_zeros(test_env.minibatch_size)
			for _ in range(100):
				_, _, rewards = learner(test_env)
				costs -= torch.stack(rewards).sum(0).squeeze(-1)
			costs = costs / 100
		else:
			_, _, rs = learner(test_env)
			costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
		mean = costs.mean()
		std = costs.std()
		gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

	print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
	return mean.item(), std.item(), gap.item()


def val_epoch(args, test_env, learner):
	learner.eval()
	with torch.no_grad():
		if args.problem_type[0] == "s":
			costs = test_env.nodes.new_zeros(test_env.minibatch_size)
			for _ in range(100):
				_, _, rewards = learner(test_env)
				costs -= torch.stack(rewards).sum(0).squeeze(-1)
			costs = costs / 100
		else:
			_, _, rs = learner(test_env)
			costs = -torch.stack(rs).sum(dim = 0).squeeze(-1)
		mean = costs.mean()
		std = costs.std()

	print("Cost on val dataset: {:5.2f} +- {:5.2f}".format(mean, std))
	return mean.item(), std.item()


def main(args):
	dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	set_random_seed(args.rng_seed, deterministic = True)

	if args.verbose:
		verbose_print = print
	else:
		def verbose_print(*args, **kwargs):
			pass

	Dataset = {
		"vrp": VRP_Dataset,
		"vrptw": VRPTW_Dataset,
		"svrptw": VRPTW_Dataset,
		"sdvrptw": SDVRPTW_Dataset,
		"dvrptw": DVRPTW_Dataset,
	}.get(args.problem_type)

	gen_params = [
		args.customers_count,
		args.vehicles_count,
		args.veh_capa,
		args.veh_speed,
		args.min_cust_count,
		args.loc_range,
		args.dem_range,
	]
	if args.problem_type != "vrp":
		gen_params.extend([args.horizon, args.dur_range, args.tw_ratio, args.tw_range])
	if args.problem_type == "sdvrptw" or args.problem_type == "dvrptw":
		gen_params.extend([args.deg_of_dyna, args.appear_early_ratio])

	verbose_print(
		"Generating {} {} samples of training data...".format(args.iter_count * args.batch_size, args.problem_type.upper()),
		end = " ",
		flush = True,
	)
	train_data = Dataset.generate(args.iter_count * args.batch_size, *gen_params)
	train_data.normalize()
	verbose_print("Done.")

	verbose_print(
		"Generating {} {} samples of test data...".format(args.test_batch_size, args.problem_type.upper()),
		end = " ",
		flush = True,
	)
	test_data = Dataset.generate(args.test_batch_size, *gen_params)
	verbose_print("Done.")

	if ORTOOLS_ENABLED:
		ref_routes = ort_solve(test_data)
	elif LKH_ENABLED:
		ref_routes = lkh_solve(test_data)
	else:
		ref_routes = None
		print("Warning! No external solver found to compute gaps for test.")
	test_data.normalize()

	Environment = {
		"vrp": VRP_Environment,
		"vrptw": VRPTW_Environment,
		"svrptw": SVRPTW_Environment,
		"sdvrptw": SDVRPTW_Environment,
		"dvrptw": DVRPTW_Environment,
	}.get(args.problem_type)

	env_params = [args.pending_cost]
	if args.problem_type != "vrp":
		env_params.append(args.late_cost)
		if args.problem_type != "vrptw" and args.problem_type != "dvrptw":
			env_params.extend([args.speed_var, args.late_prob, args.slow_down, args.late_var])

	test_env = Environment(test_data, None, None, *env_params)
	if ref_routes is not None:
		ref_costs = eval_apriori_routes(test_env, ref_routes, 100 if args.problem_type[0] == "s" else 1)
		print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
	test_env.nodes = test_env.nodes.to(dev)
	if test_env.init_cust_mask is not None:
		test_env.init_cust_mask = test_env.init_cust_mask.to(dev)

	verbose_print("Initializing attention model...", end = " ", flush = True)
	learner = COAST(
		Dataset.CUST_FEAT_SIZE,
		Environment.VEH_STATE_SIZE,
		model_size = args.model_size,
		layer_count = args.layer_count,
		head_count = args.head_count,
		ff_size = args.ff_size,
		tanh_xplor = args.tanh_xplor,
		greedy = False,
		edge_feat_size = args.edge_feat_size,
		cust_k = args.cust_k,
		memory_size = args.memory_size,
		lookahead_hidden = args.lookahead_hidden,
		dropout = args.dropout,
		adaptive_depth = args.adaptive_depth,
		adaptive_min_layers = args.adaptive_min_layers,
		adaptive_easy_ratio = args.adaptive_easy_ratio,
		latent_bottleneck = args.latent_bottleneck,
		latent_tokens = args.latent_tokens,
		latent_min_nodes = args.latent_min_nodes,
		use_edge_features = args.use_edge_features,
		use_memory = args.use_memory,
		use_ownership = args.use_ownership,
		use_lookahead = args.use_lookahead,
		fusion_mode = args.fusion_mode,
		linear_fusion_weights = args.linear_fusion_weights,
	)
	learner.to(dev)
	verbose_print("Done.")

	args.baseline_type = "critic"
	args.loss_use_cumul = False
	baseline = CriticBaseline(
		learner,
		args.customers_count,
		args.critic_use_qval,
		args.loss_use_cumul,
	)
	baseline.to(dev)

	verbose_print("Initializing PPO optimizer...", end = " ", flush = True)
	optim = Adam(
		[
			{"params": learner.parameters(), "lr": args.learning_rate, "weight_decay": args.weight_decay},
			{"params": baseline.parameters(), "lr": args.critic_rate, "weight_decay": args.weight_decay},
		]
	)
	lr_sched = None
	if args.rate_decay is not None:
		critic_decay = args.rate_decay if args.critic_decay is None else args.critic_decay
		lr_sched = LambdaLR(optim, [lambda ep: args.rate_decay**ep, lambda ep: critic_decay**ep])
	verbose_print("Done.")

	args.output_dir = (
		"./output/PPO_{}n{}m{}_{}".format(
			args.problem_type.upper(),
			args.customers_count,
			args.vehicles_count,
			time.strftime("%y%m%d-%H%M"),
		)
		if args.output_dir is None
		else args.output_dir
	)
	os.makedirs(args.output_dir, exist_ok = True)
	write_config_file(args, os.path.join(args.output_dir, "args.json"))

	if args.resume_state is None:
		start_ep = 0
	else:
		start_ep = load_checkpoint(args, learner, optim, baseline, lr_sched)

	load_model_weights(args, learner)

	verbose_print("Running PPO...")
	train_stats = []
	val_stats = []
	test_stats = []
	best_val_mu = float("inf")
	best_ep = -1

	best_ckpt_path = os.path.join(args.output_dir, "chkpt_best.pyth")
	if os.path.exists(best_ckpt_path):
		try:
			best_ckpt = torch.load(best_ckpt_path, map_location = "cpu", weights_only = False)
			loaded_best = best_ckpt.get("best_val_mu", None)
			loaded_best_ep = best_ckpt.get("best_ep", None)
			if loaded_best is not None:
				best_val_mu = float(loaded_best)
			if loaded_best_ep is not None:
				best_ep = int(loaded_best_ep)
		except Exception:
			pass

	scaler = GradScaler(enabled = args.amp)

	try:
		for ep in range(start_ep, args.epoch_count):
			train_stats.append(train_epoch(args, train_data, Environment, env_params, baseline, optim, dev, ep, scaler))
			if ref_routes is not None:
				test_stats.append(test_epoch(args, test_env, learner, ref_costs))

			val_stats.append(val_epoch(args, test_env, learner))
			cur_val_mu = val_stats[-1][0]
			if math.isfinite(cur_val_mu) and cur_val_mu < best_val_mu:
				best_val_mu = float(cur_val_mu)
				best_ep = ep
				save_best_val_checkpoint(args, ep, learner, optim, baseline, lr_sched, best_val_mu)
				verbose_print("[BEST] ep={} val_mu={:.6g} -> chkpt_best.pyth".format(ep + 1, best_val_mu))
			update_train_test_stats(args, ep, train_stats, val_stats)

			if args.rate_decay is not None:
				lr_sched.step()
			if args.pend_cost_growth is not None:
				env_params[0] *= args.pend_cost_growth
			if args.late_cost_growth is not None and len(env_params) > 1:
				env_params[1] *= args.late_cost_growth
			if args.grad_norm_decay is not None and args.max_grad_norm is not None:
				args.max_grad_norm *= args.grad_norm_decay

			if (ep + 1) % args.checkpoint_period == 0:
				save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

	except KeyboardInterrupt:
		save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
		export_train_test_stats(args, start_ep, train_stats, test_stats)
	finally:
		save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
		export_train_test_stats(args, start_ep, train_stats, test_stats)
		if best_ep >= 0:
			verbose_print("Best validation checkpoint: ep={} val_mu={:.6g}".format(best_ep + 1, best_val_mu))


if __name__ == "__main__":
	main(parse_args())
