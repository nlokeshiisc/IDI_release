# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from utils.anomaly import PotentialRootCause
import json
import networkx as nx
import os
import pandas as pd
import constants
from pathlib import Path

import numpy as np
from typing import List
import json
import random
from utils.results_processing import save_petshop_results
from utils.data_preprocessing import reduce_df


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)


def load_scenario(path):
    graph = nx.from_pandas_adjacency(
        pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
        create_using=nx.DiGraph,
    )

    # Reverse the petshop call graph to get the causal graph.
    causal_graph = graph.reverse()

    normal_metrics = pd.read_csv(
        os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0
    )

    issues = {"train": [], "test": []}
    issue_files = {"train": [], "test": []}
    for split in issues:
        issue_jsons = sorted(os.listdir(os.path.join(path, split)))
        constants.logger.warning(
            f"For path {path} and split {split}, issues in order: {issue_jsons}."
        )
        issue_files[split] = issue_jsons

        for issue in issue_jsons:
            if issue.startswith("."):  # Skip hidden files and folders.
                continue
            metrics = pd.read_csv(
                os.path.join(path, split, issue, "metrics.csv"),
                header=[0, 1, 2],
                index_col=0,
            )
            with open(os.path.join(path, split, issue, "target.json"), "r") as f:
                target = json.load(f)
            issues[split].append((metrics, target))
    return causal_graph, normal_metrics, issues, issue_files


def analyze_root_causes(
    graph: nx.DiGraph,
    target_node: str,
    target_metric: str,
    target_statistic: str,
    normal_metrics: pd.DataFrame,
    abnormal_metrics: pd.DataFrame,
) -> List[PotentialRootCause]:
    """Method to identify potential root causes that of a performance issue in target_node.

    Args:
        graph: Call graph of microservice architecture.
        target_node: Node whose SLO violoation to investigate.
        target_metric: Metric that is in violation with SLO.
        target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
        normal_metrics: Metrics of all microservices during previous normal operations.
        abnormal_metrics: Metrics of all microservices during SLO violation.

    Returns: List of potential root causes identifying nodes and assigning them scores.
    """
    raise NotImplementedError


def evaluate_specificity(analyze_root_causes, dir: str) -> pd.DataFrame:
    """Computes the fraction of normal cases for which a root-cause was detected even though there was no issue.

    Args:
        analyze_root_causes: Method that produces potential root causes for a given SLO violation.
        dir: Directory of the benchmarking dataset.

    Returns: A DataFrame with the specificity for the scenarios and targets.
    """
    scenarios = [
        "low_traffic",
        "high_traffic",
        "temporal_traffic1",
        "temporal_traffic2",
    ]
    result_list = []
    for scenario in scenarios:
        graph, normal_metrics, issues = load_scenario(os.path.join(dir, scenario))
        # We only use these issues to gather target node and metric information.
        # We do not use the abnormal metrics here.
        target_nodes_and_metrics = set(
            [
                (
                    target["target"]["node"],
                    target["target"]["metric"],
                    target["target"]["agg"],
                )
                for issues_split in issues.values()
                for (_, target) in issues_split
            ]
        )
        number_of_timestamps = len(normal_metrics.index)

        for target_node, target_metric, target_statistic in target_nodes_and_metrics:
            correct = []
            # split normal into normal and abnormal metrics
            for split in range(
                int(number_of_timestamps / 2),
                number_of_timestamps - 3,
                int(number_of_timestamps * 0.1),
            ):
                new_normal_metrics = normal_metrics.iloc[:split, :]
                abnormal_metrics = normal_metrics.iloc[split:, :]
                potential_root_causes = analyze_root_causes(
                    graph,
                    target_node,
                    target_metric,
                    target_statistic,
                    new_normal_metrics,
                    abnormal_metrics,
                )
                correct.append(len(potential_root_causes) == 0)
            row = {
                "scenario": scenario,
                "metric": target_metric,
                "ground_truth": None,
                "specificity": np.mean(correct),
            }
            result_list.append(row)
    return pd.DataFrame(result_list)


def evaluate(
    method_name, analyze_root_causes, dir: str, split: str = None
) -> pd.DataFrame:
    """Computes the top-k recall of the method to analyze root causes.

    Args:
        analyze_root_causes: Method that produces potential root causes for a given SLO violation.
        dir: Directory of the benchmarking dataset.
        split: Split of train or test, defaults to None using both.

    Returns: A DataFrame with the top-1 and top-3 recall.
    """
    scenarios = [
        "low_traffic",
        "high_traffic",
        "temporal_traffic1",
        "temporal_traffic2",
    ]
    # scenarios = ["high_traffic"]

    Path("results/petshop").mkdir(parents=True, exist_ok=True)
    constants.set_logger(dir=Path("results/petshop"), file_name=f"{method_name}.log")

    results = {}
    result_list = []
    if split is None:
        splits = ["test"]
    for scenario in scenarios:
        results[scenario] = {}

        causal_graph, normal_metrics_original, issues, issue_files = load_scenario(
            os.path.join(dir, scenario)
        )
        for split in splits:
            # set seed here for reproducibility
            set_seed()
            results[scenario][split] = {1: [], 3: []}
            for idx, (abnormal_metrics, target) in enumerate(issues[split]):
                # Slice the dataframes to only include the target metric and statistic
                target_metric = target["target"]["metric"]
                target_statistic = target["target"]["agg"]
                normal_metrics = normal_metrics_original.copy()
                normal_metrics = reduce_df(
                    normal_metrics, metric=target_metric, statistic=target_statistic
                )
                abnormal_metrics = reduce_df(
                    abnormal_metrics, metric=target_metric, statistic=target_statistic
                )

                constants.logger.info(
                    f"****" * 10 + f"{issue_files[split][idx]}" + f"****" * 10
                )
                constants.logger.info(f"Scenario: {scenario}\t Split: {split}")
                constants.logger.info(f"Target: {json.dumps(target, indent=2)}")

                kwargs = (
                    {"root_cause_nodes": [target["root_cause"]["node"]]}
                    if method_name in ["idint", "oodcf"]
                    else {}
                )
                # The above kwargs are used only to debug the idint and oodcf methods. They are not used explicitly inside the methods.
                potential_root_causes = analyze_root_causes(
                    causal_graph=causal_graph,
                    target_node=target["target"]["node"],
                    normal_metrics=normal_metrics,
                    abnormal_metrics=abnormal_metrics,
                    **kwargs,
                )

                for k in results[scenario][split]:
                    correct = in_top_k(
                        potential_root_causes,
                        ground_truth_node=target["root_cause"]["node"],
                        ground_truth_metric=target["root_cause"]["metric"],
                        k=k,
                    )
                    row = {
                        "scenario": scenario,
                        "split": split,
                        "topk": k,
                        "metric": target["target"]["metric"],
                        "issue": issue_files[split][idx],
                        "ground_truth": target["root_cause"]["node"],
                        "intopk": correct,
                        "empty": not potential_root_causes,
                    }
                    results[scenario][split][k].append(correct)
                    result_list.append(row)
                constants.logger.info(f"****" * 24)
    return pd.DataFrame(result_list)


def in_top_k(
    potential_root_causes: List[PotentialRootCause],
    ground_truth_node: str,
    ground_truth_metric: str = None,
    k: int = 3,
) -> bool:
    """Computes top-k recall of the potential root-causes in the ranked paths compared to the ground truth.

    Args:
        potential_root_causes: The potential root causes with their score.
        ground_truth: The true root cause.
        k: top-k parameter.

    Returns: True if and only if the ground truth is among the top-k potential root causes.
    """
    potential_root_causes = sorted(
        potential_root_causes, key=lambda x: x.score, reverse=True
    )
    for potential_root_cause in potential_root_causes[:k]:
        if ground_truth_node == potential_root_cause.node:
            if (
                ground_truth_metric is None
                or ground_truth_metric == potential_root_cause.metric
            ):
                return True
    return False
