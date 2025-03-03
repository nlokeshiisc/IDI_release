import os
import concurrent.futures
import argparse
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Add arguments
parser = argparse.ArgumentParser(description="Run experiments for all methods.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="syn",
    help="Dataset Name; options petshop, syn",
)
parser.add_argument(
    "--num_rcs",
    type=int,
    help="Number of root causes to inject into the synthetic test cases.",
    default=3,
)
parser.add_argument("--ocpp", type=str2bool, default=True, help="One cause per path.")
parser.add_argument(
    "--linear_eqns", type=str2bool, default=True, help="Linear equations."
)
parser.add_argument("--gpu", type=int, default=-1, help="GPU ID to use.")
parser.add_argument("--method", type=str, default="all", help="Which method to run")
parser.add_argument("--invertible", type=str2bool, default=True, help="Invertible SCMs")
args = parser.parse_args()

all_methods = [
    "traversal",
    "hierarchical_rcd",
    "epsilon_diagnosis",
    "circa",
    "ranked_correlation",
    "random_walk",
    "idint",
    "oodcf",
    "counterfactual_attribution",
    "toca",
    "smooth_traversal",
]

syn_methods = [
    "traversal",
    "circa",
    "ranked_correlation",
    "random_walk",
    "idint",
    "oodcf",
    "counterfactual_attribution",
    "toca",
    "smooth_traversal",
]

one_cause_per_path = args.ocpp
dataset = args.dataset_name
num_rcs = args.num_rcs
linear_eqns = args.linear_eqns
gpu = args.gpu
method = args.method
invertible = args.invertible

if method == "all":
    methods = all_methods if dataset == "petshop" else syn_methods
else:
    methods = [method]

occp_str = "--one_cause_per_path" if one_cause_per_path else "--no-one_cause_per_path"
linear_eqns_str = "--linear_eqns" if linear_eqns else "--no-linear_eqns"
invertible_str = "--invertible" if invertible else "--no-invertible"

def run_experiment(method):
    command = f"python run_experiments.py --dataset_name {dataset} --dataset_path dataset --method {method} --num_root_causes {num_rcs} --gpu {gpu} {occp_str} {linear_eqns_str} {invertible_str}"
    print(f"Running command: {command}")
    os.system(command)


# Using ThreadPoolExecutor to run the methods in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit all methods for parallel execution
    executor.map(run_experiment, methods)

command = f"python collate_results.py --dataset {dataset}"
os.system(command)
