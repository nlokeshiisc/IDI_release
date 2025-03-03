import pandas as pd
import os
import numpy as np
import constants


def save_petshop_results(method_name, df, verbose=False):
    """
    Save the results of the recall evaluation to a csv file.
    """
    filename = f"results/petshop/results_recall_{method_name}.csv"
    df.to_csv(filename, index=False)

    if verbose:
        for scenario in ["low_traffic", "high_traffic", "temporal_traffic"]:
            for issue_metric in ["latency", "availability"]:
                df_sel = df[df["scenario"].str.startswith(scenario)]
                df_sel = df_sel[df_sel["metric"] == issue_metric]
                for k in [1, 3]:
                    df_sel_k = df_sel[df_sel["topk"] == k]
                    size_all = len(df_sel_k.intopk)
                    res = np.mean(df_sel_k.intopk)
                    empty = len(df_sel_k[df_sel_k["empty"] == True]) / size_all
                    print(
                        f"for {scenario} with {issue_metric} with {size_all} ({empty:.2f} with no results) many issues at top{k} got {res:.3f}"
                    )


def save_syn_results(method_name, df: pd.DataFrame, verbose=False):
    """Saves the results for syn Experiments

    Args:
        method_name (_type_): _description_
        df (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
    """
    filename = f"results_recall_{method_name}.csv"
    df.to_csv(constants.RESULTS_DIR / filename, index=False)
