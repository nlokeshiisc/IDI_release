# %%
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Run experiments for all methods.")
parser.add_argument(
    "--dataset",
    type=str,
    default="syn",
    help="Dataset Name; options petshop, syn",
)
args = parser.parse_args()
dataset = args.dataset

if dataset == "petshop":
    results_dir = Path("results/petshop")

    # %%
    files = list(results_dir.glob("*.csv"))
    collated_results = defaultdict(list)

    scenarios = ["low", "high", "temporal"]
    metric = ["latency", "availability"]
    topk = [1, 3]

    for s in scenarios:
        for m in metric:
            try:
                for k in topk:
                    collated_results["scenario"].append(s)
                    collated_results["metric"].append(m)
                    collated_results["topk"].append(k)
            except:
                pass

    # %%
    for file in files:
        name = file.stem.split("results_recall_")[-1]
        if "recall" not in file.name:
            continue
        df = pd.read_csv(file)

        for s in scenarios:
            for m in metric:
                for k in topk:
                    # fmt: off
                    filtered_df = df[df["metric"] == m][df["topk"] == k][df["split"] == "test"]
                    sfilter = [s in x for x in filtered_df["scenario"]]
                    filtered_df = filtered_df[sfilter]
                    acc = (filtered_df["intopk"].values * 1).mean()
                    collated_results[name].append(acc)
                    # fmt: on
    # %%
    pd.DataFrame(collated_results).to_csv(
        results_dir / "collated_results.csv", index=False
    )

elif dataset == "syn":
    for sfx in ["_linear", "_nonlinear"]:
        root_dir = Path(f"results/syn{sfx}")
        for num_rcs in [1, 3]:
            subdir = root_dir / f"rc-{num_rcs}"
            for ocpp in ["True", "False"]:
                ocpp_dir = subdir / f"ocpp-{ocpp}"

                for invertible in [None, "non-invertible"]:
                    if invertible is not None:
                        ocpp_dir = ocpp_dir / invertible

                    if not ocpp_dir.exists():
                        continue

                    csv_files = list(ocpp_dir.glob("*.csv"))
                    collated_results = defaultdict(list)
                    collated_results["method"] = []
                    collated_results["Top-1"] = []
                    collated_results["Top-3"] = []

                    for file in csv_files:
                        if "recall" not in file.name:
                            continue
                        method_name = file.stem.split("results_recall_")[-1]
                        collated_results["method"].append(method_name)
                        df = pd.read_csv(file)
                        for k in [1, 3]:
                            df_k = df[df["k"] == k]
                            acc = (df_k["Recall"].values * 1).mean()
                            collated_results[f"Top-{k}"].append(acc)

                    pd.DataFrame(collated_results).to_csv(
                        ocpp_dir / "collated_results.csv", index=False
                    )
