#!/usr/bin/env python3
"""Small test utility to exercise `Problem.tensor_load`.

Usage example:
  python -m python_src.tools.test_problem_tensor_load --file datasets/dvrptw_n100m5_1000.pyth --index 1 --truck-speed 1.0 --truck-capacity 1300.0 --num-trucks 5
"""
import argparse
import sys

from python_src.sim.problem import Problem


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="path to torch dataset file")
    p.add_argument("--index", type=int, default=0, help="instance index in the batch to load")
    p.add_argument("--truck-speed", type=float, default=1.0)
    p.add_argument("--truck-capacity", type=float, default=200.0)
    p.add_argument("--num-trucks", type=int, default=5)
    return p.parse_args(argv)


def main():
    args = parse_args()
    try:
        problem = Problem.tensor_load(args.file, instance_num=args.index, truck_speed=args.truck_speed, truck_capacity=args.truck_capacity, num_trucks=args.num_trucks)
    except Exception as e:
        print("Failed to load problem:", e)
        sys.exit(2)

    print("Loaded Problem")
    depot = problem.depot
    print(f" Depot idx={depot.idx} x={depot.x} y={depot.y} close={depot.close}")
    print(f" Num requests (excluding depot): {len(problem.requests)}")
    print(" First 5 requests:")
    for i, r in enumerate(problem.requests[:5]):
        print(f"  {i+1}: idx={r.idx} x={r.x} y={r.y} dem={r.demand} open={r.open} close={r.close} dur={r.service_time} time={r.time}")


if __name__ == "__main__":
    main()
