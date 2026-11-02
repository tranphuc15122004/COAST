"""Genetic Programming helpers (GPContext) - Python port of `gp::mod.rs`.

Responsibilities:
- Provide `GPContext` which implements generation strategies (full/grow),
	mutation, crossover, and helper utilities for program trees.
- Drive population initialization (`ramp_half_and_half`) and provide
	operations used by the GP loop in `main.py`.

Important methods on `GPContext`:
- `gen_full_at`, `gen_grow_at`, `gen_generic_at`: program tree builders.
- `mutation(program)`: perform subtree mutation.
- `crossover(p1, p2)`: single-point crossover by subtree swapping.
- `ramp_half_and_half(context)`: build an initial population mixing full
	and grow trees (returns list of `Program`).

Notes:
- This module uses the `Program` API from `gp.program` and expects a
	`program.context` to expose the ProgramContext-like interface when
	generation relies on context-specific arities/terminals.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Callable, Iterable, List, Tuple

try:
	from ..main import CONST_RATE  # best-effort import if exposed
except Exception:
	CONST_RATE = 0.1

from .GPtree import Node, Program, ProgramContext, MAX_PROGRAM_NODE_CHILDREN


# chứa các hàm tiện ích trong quá trình sử dụng cây GP
class GPContext:
	def __init__(self, rng: random.Random, num_population: int, max_depth: int):
		self.rng = rng
		self.num_population = num_population
		self.max_depth = max_depth

	def gen_terminal_at(self, program: Program, index: int) -> None:
		"""Generate a terminal at `index` (or a small internal chunk with probability CONST_RATE)."""
		terminal = self.rng.random() < (1.0 - CONST_RATE)
		if terminal:
			term_index = self.rng.randrange(0, program.context.num_terminals())
			program.generate_at(index, 0, Node.Terminal(term_index), lambda *_: None)
		else:
			# fall back to generate a small (internal like) value — keep behavior flexible
			raw = self.rng.randrange(0, 9) * 16
			program.generate_at(index, 0, raw, lambda *_: None)

	def gen_internal_at(self, program: Program, index: int, gen_child_fn: Callable) -> None:
		int_index = self.rng.randrange(0, program.context.num_internals())
		num_children = program.context.internal_num_children(int_index)
		program.generate_at(index, num_children, Node.Internal(int_index), gen_child_fn)

	def gen_generic_at(self, program: Program, index: int, gen_child_fn: Callable) -> None:
		type_index = self.rng.randrange(0, program.context.num_internals() + program.context.num_terminals())
		if type_index < program.context.num_terminals():
			value = Node.Terminal(type_index)
			num_children = 0
		else:
			ii = type_index - program.context.num_terminals()
			value = Node.Internal(ii)
			num_children = program.context.internal_num_children(ii)
		program.generate_at(index, num_children, value, gen_child_fn)

	def gen_full_at(self, program: Program, index: int, depth: int) -> None:
		if depth == 0:
			self.gen_terminal_at(program, index)
		else:
			def child_gen(p, _, child_index):
				self.gen_full_at(p, child_index, depth - 1)

			self.gen_internal_at(program, index, child_gen)

	def gen_grow_at(self, program: Program, index: int, depth: int) -> None:
		if depth == 0:
			self.gen_terminal_at(program, index)
		else:
			def child_gen(p, _, child_index):
				self.gen_grow_at(p, child_index, depth - 1)

			self.gen_generic_at(program, index, child_gen)

	@staticmethod
	def depth_from_top(index: int) -> int:
		# replicates (index + 1).ilog2() from Rust
		return (index + 1).bit_length() - 1

	def depth_to_bottom(self, program: Program, index: int) -> int:
		node = Node.from_value(program.nodes[index])
		if node.is_internal():
			num_children = program.context.internal_num_children(node.internal_index())
			depths = []
			for child_index in Program.child_indices(index, num_children):
				depths.append(self.depth_to_bottom(program, child_index) + 1)
			return max(depths) if depths else 0
		return 0

	def mutation(self, p: Program) -> Program:
		p = copy.deepcopy(p)
		indices = list(p.all_active_indices())
		swap_pos = self.rng.choice(indices)
		p.clear_subtree(swap_pos)
		depth = self.max_depth - GPContext.depth_from_top(swap_pos)
		self.gen_grow_at(p, swap_pos, depth)
		p.verify()
		return p

	@staticmethod
	def copy_subtree(dest: Program, dest_index: int, src: Program, src_index: int) -> None:
		node = Node.from_value(src.nodes[src_index])
		if node.is_internal():
			# use the source program's context to determine arity
			num_children = src.context.internal_num_children(node.internal_index()) if src.context else 0
		else:
			num_children = 0

		def gen_child(d, i, d_child_index):
			GPContext.copy_subtree(d, d_child_index, src, src_index * MAX_PROGRAM_NODE_CHILDREN + i + 1)

		dest.generate_at(dest_index, num_children, src.nodes[src_index], gen_child)

	@staticmethod
	def all_index_of_layer(layer: int) -> range:
		return range((1 << layer) - 1, (1 << (layer + 1)) - 1)

	def crossover(self, p1: Program, p2: Program) -> Tuple[Program, Program]:
		c1 = copy.deepcopy(p1)
		c2 = copy.deepcopy(p2)
		depth1 = self.depth_to_bottom(c1, 0)
		depth2 = self.depth_to_bottom(c2, 0)

		depth_point1 = self.rng.randrange(0, depth1 + 1)
		min_depth_point2 = max(0, depth_point1 + depth2 - self.max_depth)
		max_depth_point2 = min(depth2, self.max_depth - depth1 + depth_point1)

		depth_point2 = self.rng.randrange(min_depth_point2, max_depth_point2 + 1)

		def choose_swap_idx(c, depth_point):
			candidates = [i for i in GPContext.all_index_of_layer(depth_point)
						  if i < len(c.nodes) and not Node.from_value(c.nodes[i]).is_null()]
			if not candidates:
				raise RuntimeError("no valid swap index found")
			return self.rng.choice(candidates)

		swap_idx1 = choose_swap_idx(c1, depth_point1)
		swap_idx2 = choose_swap_idx(c2, depth_point2)

		c1.clear_subtree(swap_idx1)
		c2.clear_subtree(swap_idx2)

		GPContext.copy_subtree(c1, swap_idx1, p2, swap_idx2)
		GPContext.copy_subtree(c2, swap_idx2, p1, swap_idx1)

		assert self.depth_to_bottom(c1, 0) <= self.max_depth
		assert self.depth_to_bottom(c2, 0) <= self.max_depth

		c1.verify()
		c2.verify()

		return c1, c2

	def ramp_half_and_half(self, context=None) -> List[Program]:
		v: List[Program] = []
		half_size = self.num_population // 2
		for depth in range(1, self.max_depth):
			for _ in range(half_size // max(1, self.max_depth)):
				p = Program.new(context=context)
				self.gen_full_at(p, 0, depth)
				p.verify()
				v.append(p)
				p2 = Program.new(context=context)
				self.gen_grow_at(p2, 0, depth)
				p2.verify()
				v.append(p2)

		while len(v) < self.num_population:
			p = Program.new(context=context)
			self.gen_grow_at(p, 0, self.max_depth)
			p.verify()
			v.append(p)

		return v
