"""Program / Node representation for GP (Python port of gp::program.rs).

Responsibilities:
- Define `Node` (const/terminal/internal/null) and encode/decode between
    compact byte values and Node instances.
- Implement `Program`, a flat-array representation of a tree with helpers:
    - evaluation: `calc(ctx)` using a supplied ProgramContext-like object
    - formatting: `fmt(ctx)` to pretty-print expressions
    - encoding: run-length encode / base64 (`base64()` / `from_base64`)
    - utilities: subtree collect/clear, verify, and RLE helpers.

Program expects a context object (or class) with methods like
`num_terminals()`, `num_internals()`, `internal_num_children(i)`,
`terminal(i)`, `internal(i, child_values)`. See `sim.ctx` for concrete
`RoutingContext`/`SequencingContext` implementations.
"""
from __future__ import annotations

import base64
from typing import Callable, Iterable, List

MAX_PROGRAM_NODE_CHILDREN = 2


class Node:
    def __init__(self, kind: str, value=None):
        self.kind = kind  # 'const', 'terminal', 'internal', 'null'
        self.value = value

    @staticmethod
    def Const(value: float) -> "Node":
        return Node("const", float(value))

    @staticmethod
    def Terminal(index: int) -> "Node":
        return Node("terminal", int(index))

    @staticmethod
    def Internal(index: int) -> "Node":
        return Node("internal", int(index))

    @staticmethod
    def Null() -> "Node":
        return Node("null", None)

    @staticmethod
    def from_value(x: int) -> "Node":
        if 0 <= x <= 128:
            # const: (x as i16 - 64) / 16.0
            val = (int(x) - 64) / 16.0
            return Node("const", val)
        if 129 <= x <= 192:
            return Node("terminal", x - 129)
        if 193 <= x <= 254:
            return Node("internal", x - 193)
        return Node("null", None)

    def to_value(self) -> int:
        if self.kind == "const":
            return int(self.value * 16.0 + 64.0) & 0xFF
        if self.kind == "terminal":
            return 129 + int(self.value)
        if self.kind == "internal":
            return 193 + int(self.value)
        return 255

    def is_null(self) -> bool:
        return self.kind == "null"

    def is_internal(self) -> bool:
        return self.kind == "internal"

    def internal_index(self) -> int:
        return int(self.value)

    def __repr__(self) -> str:
        if self.kind == "const":
            return f"Const({self.value})"
        if self.kind == "terminal":
            return f"Terminal({self.value})"
        if self.kind == "internal":
            return f"Internal({self.value})"
        return "Null"


# abstract class cho hai loại cây ROUTING và SEQUENCING
class ProgramContext:
    """Lightweight placeholder for the ProgramContext API expected by GP code.

    Concrete contexts should implement the instance methods used by `Program.calc`:
      - num_terminals()
      - num_internals()
      - internal_num_children(index)
      - terminal(i)
      - internal(index, child_values)
      - (optional) format_terminal / format_internal
    """
    @staticmethod
    def num_terminals() -> int:
        raise NotImplementedError()

    @staticmethod
    def num_internals() -> int:
        raise NotImplementedError()

    @staticmethod
    def internal_num_children(index: int) -> int:
        raise NotImplementedError()

    def terminal(self, index: int) -> float:
        raise NotImplementedError()

    def internal(self, index: int, child_values):
        raise NotImplementedError()


# CÂY GP
class Program:
    def __init__(self, nodes: List[int] = None):
        self.nodes: List[int] = list(nodes) if nodes is not None else []
        # optional context object expected to implement the ProgramContext-like API
        self.context = None

    @classmethod
    def new(cls, context=None) -> "Program":
        p = cls([])
        p.context = context
        return p

    @classmethod
    def from_vec(cls, nodes: List[int]) -> "Program":
        # Accept nodes as ints or `Node` instances; convert to raw byte values.
        conv: List[int] = []
        for x in nodes:
            if isinstance(x, Node):
                conv.append(x.to_value())
            else:
                conv.append(int(x) & 0xFF)
        return cls(conv)

    @classmethod
    def terminal(cls, index: int) -> "Program":
        p = cls([Node("terminal", index).to_value()])
        return p

    @staticmethod
    def child_indices(index: int, num_children: int) -> Iterable[int]:
        for i in range(num_children):
            yield index * MAX_PROGRAM_NODE_CHILDREN + i + 1

    def ensure_index_exist(self, index: int) -> None:
        if index >= len(self.nodes):
            self.nodes.extend([255] * (index + 1 - len(self.nodes)))

    def generate_at(self, index: int, num_children: int, value, gen_child_fn: Callable[["Program", int, int], None]) -> None:
        # value may be Node or raw int
        self.ensure_index_exist(index)
        if isinstance(value, Node):
            b = value.to_value()
        else:
            b = int(value) & 0xFF
        self.nodes[index] = b
        for i, child_index in enumerate(Program.child_indices(index, num_children)):
            gen_child_fn(self, i, child_index)

    def calc_at(self, ctx, i: int, term_cache: List[float]) -> float:
        node = Node.from_value(self.nodes[i])
        if node.kind == "const":
            return float(node.value)
        if node.kind == "terminal":
            return float(term_cache[int(node.value)])
        if node.kind == "internal":
            num_children = ctx.internal_num_children(int(node.value))
            children = [self.calc_at(ctx, ci, term_cache) for ci in Program.child_indices(i, num_children)]
            return float(ctx.internal(int(node.value), children))
        raise RuntimeError("Null node encountered in calc_at")

    def calc(self, ctx) -> float:
        term_cache = [ctx.terminal(i) for i in range(ctx.num_terminals())]
        return self.calc_at(ctx, 0, term_cache)

    def collect_all_active_indices(self, dest: List[int], index: int) -> None:
        dest.append(index)
        node = Node.from_value(self.nodes[index])
        if node.is_internal():
            for child_index in Program.child_indices(index, self.context.internal_num_children(node.internal_index()) if self.context else 0):
                self.collect_all_active_indices(dest, child_index)

    def all_active_indices(self) -> List[int]:
        if not self.nodes:
            return []
        indices: List[int] = []
        self.collect_all_active_indices(indices, 0)
        return indices

    @staticmethod
    def run_length_encode(v: List[int]) -> List[int]:
        res: List[int] = []
        for byte in v:
            if len(res) < 2 or res[-2] != byte or res[-1] == 255:
                res.append(byte)
                res.append(0)
            else:
                res[-1] += 1
        return res

    @staticmethod
    def run_length_decode(v: List[int]) -> List[int]:
        assert len(v) % 2 == 0
        res: List[int] = []
        for i in range(len(v) // 2):
            byte = v[i * 2]
            count = v[i * 2 + 1] + 1
            res.extend([byte] * count)
        return res

    def base64(self) -> str:
        enc = bytes(Program.run_length_encode(self.nodes))
        return base64.b64encode(enc).decode("ascii")

    @classmethod
    def from_base64(cls, s: str) -> "Program":
        raw = base64.b64decode(s)
        nodes = Program.run_length_decode(list(raw))
        return cls(nodes)

    def verify(self) -> None:
        all_actives = self.all_active_indices()
        for idx, val in enumerate(self.nodes):
            is_null = Node.from_value(val).is_null()
            is_active = idx in all_actives
            assert (is_active != is_null), f"verify failed at {idx}: active={is_active} null={is_null}"

    def clear_subtree(self, index: int) -> None:
        indices: List[int] = []
        self.collect_all_active_indices(indices, index)
        for i in indices:
            self.nodes[i] = Node("null").to_value()

    def _format_node(self, ctx, index: int) -> str:
        node = Node.from_value(self.nodes[index])
        if node.kind == "const":
            return str(node.value)
        if node.kind == "terminal":
            if hasattr(ctx, "format_terminal"):
                return ctx.format_terminal(int(node.value))
            return f"TERM{int(node.value)}"
        if node.kind == "internal":
            s = ctx.format_internal(int(node.value)) if hasattr(ctx, "format_internal") else f"INT{int(node.value)}"
            children = [self._format_node(ctx, ci) for ci in Program.child_indices(index, ctx.internal_num_children(int(node.value)))]
            return f"{s}({', '.join(children)})"
        return "INVALID"

    def fmt(self, ctx) -> str:
        if not self.nodes:
            return ""
        return self._format_node(ctx, 0)

    def __str__(self) -> str:
        # best-effort: if a context is attached, use it
        if self.context:
            try:
                return self.fmt(self.context)
            except Exception:
                pass
        # fallback: show raw nodes as Node reprs
        return str([Node.from_value(b) for b in self.nodes])
