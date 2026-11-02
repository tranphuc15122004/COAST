import argparse

import torch
import os
import random
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate normalized dataset")

    parser.add_argument("--problem-type", "-p", type=str,
                        choices=["vrp", "vrptw", "svrptw", "sdvrptw", "dvrptw"],
                        default="vrptw")
    parser.add_argument("--customers-count", "-n", type=int, default=10)
    parser.add_argument("--vehicles-count", "-m", type=int, default=2)
    parser.add_argument("--veh-capa", type=int, default=200)
    parser.add_argument("--veh-speed", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=480)
    parser.add_argument("--min-cust-count", type=int, default=None)
    parser.add_argument("--loc-range", type=int, nargs=2, default=(0,101))
    parser.add_argument("--dem-range", type=int, nargs=2, default=(5,41))
    parser.add_argument("--dur-range", type=int, nargs=2, default=(10,31))
    parser.add_argument("--tw-ratio", type=float, nargs='*', default=(0.25,0.5,0.75,1.0))
    parser.add_argument("--tw-range", type=int, nargs=2, default=(30,91))
    parser.add_argument("--deg-of-dyna", type=float, nargs='*', default=(0.1,0.25,0.5,0.75))
    parser.add_argument("--appear-early-ratio", type=float, nargs='*', default=(0.0,0.5,0.75,1.0))

    parser.add_argument("--iter-count", "-i", type=int, default=1000,
                        help="number of iterations (multiplied by batch-size to get total samples)")
    parser.add_argument("--batch-size", "-b", type=int, default=512,
                        help="mini-batch size used to compute total samples")
    parser.add_argument("--rng-seed", type=int, default=15122004)
    parser.add_argument("--output-dir", type=str, default='data')
    parser.add_argument("--load", type=str, default=None,
                        help="path to existing dataset to load and print summary (skips generation)")
    parser.add_argument("--show-index", type=int, default=None,
                        help="when loading a dataset, print the i-th scenario details")

    return parser.parse_args(argv)





def read_dataset(fpath):
    """Load a dataset file saved with torch.save and return the object."""
    try:
        return torch.load(fpath)
    except Exception as e:
        # PyTorch 2.6+ may restrict globals by default (weights_only=True).
        # If the file contains custom classes (pickled objects) we need to
        # allowlist those globals before loading with weights_only=False.
        try:
            print("Standard torch.load failed, retrying with weights_only=False (trusted file)...")
            # Attempt to import a commonly used dataset class and allowlist it
            try:
                # If your dataset was created from a module named
                # `problems._data_dtw.DVRPTW_Dataset`, import it so we can
                # add it to the safe globals. Adjust this import if your
                # dataset class lives in a different module.
                from problems._data_dtw import DVRPTW_Dataset 
                torch.serialization.add_safe_globals([DVRPTW_Dataset])
                return torch.load(fpath, weights_only=False)
            except Exception:
                # If we can't import the original class, still attempt a
                # trusted load but raise a clearer error explaining next
                # steps.
                try:
                    return torch.load(fpath, weights_only=False)
                except Exception:
                    raise RuntimeError(
                        "Failed to load file with weights_only=False and could not "
                        "allowlist the original dataset class. If you trust the "
                        "file, import the dataset class and call "
                        "torch.serialization.add_safe_globals([ThatClass]) before "
                        "loading, or re-create the dataset as a plain dict/tensors."
                    ) from None
        except Exception:
            # Re-raise original error for visibility
            raise e


def print_dataset_summary(ds):
    """Print a short summary for a loaded dataset (or dict)."""
    if hasattr(ds, 'nodes'):
        print("Loaded dataset instance:")
        try:
            print(" - batch_size:", getattr(ds, 'batch_size', None))
            print(" - nodes shape:", tuple(ds.nodes.size()))
            print(" - veh_count:", getattr(ds, 'veh_count', None))
            print(" - veh_capa:", getattr(ds, 'veh_capa', None))
        except Exception:
            pass
    elif isinstance(ds, dict):
        print("Loaded dict from file, keys:", list(ds.keys()))
        if 'nodes' in ds:
            try:
                print(" - nodes shape:", tuple(ds['nodes'].size()))
            except Exception:
                pass
    else:
        print("Loaded object of type:", type(ds))


def print_scenario(ds, idx):
    """Print detailed info for scenario index `idx` (0-based) from dataset object."""
    if hasattr(ds, 'nodes'):
        nodes = ds.nodes
        batch_size = getattr(ds, 'batch_size', nodes.size(0))
        if idx < 0 or idx >= batch_size:
            print(f"Index {idx} out of range (batch_size={batch_size})")
            return
        inst = nodes[idx]
        n_nodes, feat = inst.size()
        print(f"Scenario {idx}: nodes={n_nodes}, feat={feat}")
        depot = inst[0]
        print("Depot:")
        print(f"  x={depot[0].item():.4g}, y={depot[1].item():.4g}")
        if feat >= 5:
            print(f"  horizon/due={depot[4].item():.4g}")
        print("Customers:")
        for j in range(1, n_nodes):
            row = inst[j]
            out = [f"#{j}"]
            out.append(f"x={row[0].item():.4g}")
            out.append(f"y={row[1].item():.4g}")
            if feat >= 3:
                out.append(f"dem={row[2].item():.4g}")
            if feat >= 6:
                out.append(f"ready={row[3].item():.4g}")
                out.append(f"due={row[4].item():.4g}")
                out.append(f"dur={row[5].item():.4g}")
            if feat == 7:
                out.append(f"appear={row[6].item():.4g}")
            print("  ", ", ".join(out))
        if getattr(ds, 'cust_mask', None) is not None:
            try:
                mask = ds.cust_mask[idx]
                print("Customer mask (True=hidden):", mask.tolist())
            except Exception:
                pass
    elif isinstance(ds, dict) and 'nodes' in ds:
        nodes = ds['nodes']
        try:
            inst = nodes[idx]
        except Exception:
            print(f"Index {idx} out of range or nodes not indexable")
            return
        print(f"Scenario {idx} from dict, nodes shape: {tuple(nodes.size())}")
        print(inst)
    else:
        print("Can't print scenario for object of type", type(ds))



def main(args):
    if args.load is not None:
        ds = read_dataset(args.load)
        if args.show_index is None:
            print_dataset_summary(ds)
        else:
            print_dataset_summary(ds)
            print_scenario(ds, args.show_index)
        return

if __name__ == "__main__":
    main(parse_args())
