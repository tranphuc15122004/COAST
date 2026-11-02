#!/usr/bin/env python3
"""Utility to decode GP Program base64 encodings and pretty-print them.

Usage: `decode_base64.py <routing_base64> <sequencing_base64>`

This script decodes the run-length/base64 encoded programs produced by the
GP run and attempts to format them using the `sim.ctx` contexts so they are
readable expressions.
"""
import sys
from python_src.gp.GPtree import Program
from python_src.sim import ctx as sim_ctx

if len(sys.argv) < 3:
    print("usage: decode_base64.py <routing_base64> <sequencing_base64>")
    sys.exit(1)

r_b64 = sys.argv[1]
s_b64 = sys.argv[2]

r = Program.from_base64(r_b64)
s = Program.from_base64(s_b64)
print("Routing:" )
try:
    print(r.fmt(sim_ctx.RoutingContext))
except Exception:
    print(str(r))
print()
print("Sequencing:")
try:
    print(s.fmt(sim_ctx.SequencingContext))
except Exception:
    print(str(s))
