# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from enum import Enum
from typing import Dict, List, Any, Tuple, Union, Callable, Optional, Set
import os
import warnings

from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig

import networkx as nx
import numpy as np
import pandas as pd

from utils.anomaly import PotentialRootCause
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from utils.data_preprocessing import (
    marginalize_node,
    impute_df,
)
from utils.anomaly import get_anomalous_metrics_and_scores

from constants import AnomalyDetectionConfig, DEFAULT_ANOMALY_DETECTION


def make_random_walk(
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    imputation_method: str = "mean",
    use_partial_corr: bool = False,
    rho: float = 0.1,
    num_steps: int = 10,
    num_repeats: int = 1000,
    root_cause_top_k: int = 3,
    search_for_anomaly: bool = True,
):
    """
    Wrapper around the 'random walk' RCA method as implemented in https://github.com/salesforce/PyRCA.

    Paper: https://doi.org/10.1109/CCGRID.2018.00076, https://doi.org/10.1145/3442381.3449905

    Args:
        use_partial_corr: Whether to use partial correlation when computing edge weights.
            Deafult: False
        rho: The weight from a "cause" node to a "result" node. Default: 0.1
        num_steps: The number of random walk steps in each run. Default: 10
        num_repeats: The number of random walk runs. Default: 1000
        root_cause_top_k: The maximum number of root causes in the results. Default: 3
        search_for_anomaly: A boolean indicating whether to search for anomalies in the causal graph. Default: True
    """

    def analyze_root_causes(
        causal_graph: nx.DiGraph,
        target_node: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """Method to identify potential root causes of the performance issue in target_node.

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violoation to investigate.
            target_metric: Metric that is in violation with SLO.
            target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """
        normal_metrics = normal_metrics.copy()
        abnormal_metrics = abnormal_metrics.copy()

        normal_metrics = normal_metrics.loc[
            :, normal_metrics.columns[~normal_metrics.isna().all()]
        ]

        abnormal_metrics = abnormal_metrics.loc[
            :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
        ]

        common_cols = normal_metrics.columns.intersection(abnormal_metrics.columns)
        normal_metrics = normal_metrics.loc[:, common_cols]
        abnormal_metrics = abnormal_metrics.loc[:, common_cols]

        impute_df(normal_metrics, imputation_method)
        impute_df(abnormal_metrics, imputation_method)

        missing_nodes = normal_metrics.columns.symmetric_difference(
            causal_graph.nodes()
        )

        for node in missing_nodes:
            try:
                marginalize_node(causal_graph, node)
            except:
                pass

        # Remove missing_nodes from normal and abnormal metrics if they exist
        normal_metrics = normal_metrics.loc[
            :, normal_metrics.columns.intersection(causal_graph.nodes())
        ]
        abnormal_metrics = abnormal_metrics.loc[
            :, abnormal_metrics.columns.intersection(causal_graph.nodes())
        ]

        all_anomalous_nodes_with_score = get_anomalous_metrics_and_scores(
            normal_metrics,
            abnormal_metrics,
            target_node,
            anomaly_detection_config,
        )
        if all_anomalous_nodes_with_score == ({}, []):
            return []
        anomalous_nodes = list(all_anomalous_nodes_with_score.keys())

        for node in anomalous_nodes:
            if node not in causal_graph.nodes():
                warnings.warn(
                    f"Node {node} is not present in the causal graph. Removing it from the list of anomalous nodes."
                )
                anomalous_nodes.remove(node)

        causal_graph = nx.to_pandas_adjacency(causal_graph)

        model = RandomWalk(
            config=RandomWalkConfig(
                causal_graph,
                use_partial_corr,
                rho,
                num_steps,
                num_repeats,
                root_cause_top_k,
            )
        )
        abnormal_metrics = abnormal_metrics + np.random.normal(
            0, 0.01, abnormal_metrics.shape
        )
        result = model.find_root_causes(anomalous_nodes, abnormal_metrics)

        potential_root_causes = [
            PotentialRootCause(root_cause, score)
            for root_cause, score in result.root_cause_nodes
        ]
        return potential_root_causes

    return analyze_root_causes
