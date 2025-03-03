# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List, Union, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.whatif import interventional_samples
from dowhy.gcm.graph import get_ordered_predecessors
from utils.anomaly import train_anomaly_detectors
from constants import (
    AnomalyDetectionConfig,
    DEFAULT_ANOMALY_DETECTION,
)
from utils.data_preprocessing import impute_df, marginalize_node
from utils.scm_utils import train_scm
from utils.anomaly import PotentialRootCause
from constants import logger

warnings.filterwarnings("ignore")

gcm.config.disable_progress_bars()


def pre_process(
    *,
    causal_graph,
    target_node,
    normal_metrics,
    abnormal_metrics,
    anomaly_detection_config,
    linear_eqns,
):
    """
    Builds the necessary models required for assessing the fix and anomaly condition later.
    """

    # %% 1. Train the g function for anomaly detection.
    anomaly_detectors = train_anomaly_detectors(
        normal_metrics,
        anomaly_detection_config,
    )

    # %%2. Preprocess the dataframes and impute them.
    normal_metrics = normal_metrics.loc[
        :, normal_metrics.columns[~normal_metrics.isna().all()]
    ]
    abnormal_metrics = abnormal_metrics.loc[
        :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
    ]

    # %% 3. drop columns with all NaNs and those that did not qualify to have an anomaly detector.
    common_cols = normal_metrics.columns.intersection(abnormal_metrics.columns)
    common_cols = common_cols.intersection(anomaly_detectors.keys())

    normal_metrics = normal_metrics.loc[:, common_cols]
    abnormal_metrics = abnormal_metrics.loc[:, common_cols]

    # Impute the missing values
    impute_df(normal_metrics)
    impute_df(abnormal_metrics)

    # %% 3. Remove nodes that are not in the causal graph.
    missing_nodes = normal_metrics.columns.symmetric_difference(causal_graph.nodes())
    for node in missing_nodes:
        if node in causal_graph.nodes():
            marginalize_node(causal_graph, node)

    # %% 4. Get the Trigger point for anomaly detection.
    target_anomaly_scores = anomaly_detectors[target_node][0].score(
        abnormal_metrics[target_node].to_numpy()
    )
    trigger_step = np.argmax(target_anomaly_scores)

    abnormal_metrics = abnormal_metrics.iloc[[trigger_step]]
    target_an_score = anomaly_detectors[target_node][0].score(
        abnormal_metrics[target_node].to_numpy()
    )

    abnormaldf_an_scores = {
        node: anomaly_detectors[node][0].score(abnormal_metrics[node].to_numpy())
        for node in abnormal_metrics.columns
    }

    # %% 5. Fit an SCM for performing interventions.
    # Train a structural causal model on normal data for the target_metric and average statistic.
    learned_scm: gcm.ProbabilisticCausalModel = train_scm(
        causal_graph.copy(),
        normal_metrics,
        linear_eqns=linear_eqns,
    )
    # evaluate_scm(scm, normal_metrics)
    # print_scm_parameters(scm, normal_metrics)

    return (
        normal_metrics,
        abnormal_metrics,
        anomaly_detectors,
        target_an_score,
        abnormaldf_an_scores,
        learned_scm,
    )


def make_toca(
    n_jobs: int = -1,
    attribute_mean_deviation: bool = True,
    anomaly_scorer: gcm.anomaly_scorers.AnomalyScorer = gcm.anomaly_scorers.MeanDeviationScorer(),
    filter_for_anomalous_metrics: bool = True,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    cf_rca: bool = False,
    linear_eqns: bool = True,
):
    """Maker of a method to use counterfactuals for root-cause analysis.

    Args:
        n_jobs: use 1 for sequential computation, -1 for parallel.
        attribute_mean_deviation: Indicator whether the contribution is based on the feature relevance with respect
         to the given scoring function or the IT score.
        anomaly_scorer: Anomaly Scorer.
    """
    an_thres = anomaly_detection_config.anomaly_score_threshold
    p_an_thresh = anomaly_detection_config.parent_anomaly_score_thresold

    def analyze_root_causes(
        causal_graph: nx.DiGraph,
        target_node: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """This method implements the TOCA algorithm
        A node is a potential root cause if the following holds:
            The node itself is anomalous
            The parents are not anomalous
        It performs interventions therein to test if the potential node can fix the anomaly at the target node.

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violoation to investigate.
            target_metric: Metric that is in violation with SLO.
            target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """
        (
            normal_metrics,
            abnormal_metrics,
            anomaly_detectors,
            target_an_score,
            abnormaldf_an_scores,
            learned_scm,
        ) = pre_process(
            causal_graph=causal_graph,
            target_node=target_node,
            normal_metrics=normal_metrics,
            abnormal_metrics=abnormal_metrics,
            anomaly_detection_config=anomaly_detection_config,
            linear_eqns=linear_eqns,  # Should I learn a linear SCM or do automl?
        )

        # Get the Topological order of the ancestors of the target node.
        ancestors = list(nx.ancestors(causal_graph, target_node))
        ancestors_sorted = list(
            nx.topological_sort(causal_graph.subgraph(ancestors + [target_node]))
        )
        ancestors_sorted = ancestors_sorted[::-1]

        def _est_fix_nodes(alpha: set):
            int_dict = {}
            for node in alpha:
                parents = get_ordered_predecessors(causal_graph, node)
                if len(parents) == 0:
                    int_value = normal_metrics[node].mean()
                else:
                    node_mechanism = learned_scm.causal_mechanism(node)
                    int_value = node_mechanism.draw_samples(
                        abnormal_metrics[parents].to_numpy()
                    ).mean()
                int_dict[node] = int_value
            return int_dict

        int_nodes = _est_fix_nodes(ancestors_sorted)

        flip_counts = {node: 0 for node in ancestors_sorted}
        rc_scores = {}
        for i in range(len(ancestors_sorted)):
            int_dict = {
                ancestors_sorted[_]: lambda node: int_nodes[ancestors_sorted[_]]
                for _ in range(i + 1)
            }
            int_samples = interventional_samples(
                causal_model=learned_scm,
                interventions=int_dict,
                observed_data=pd.concat([abnormal_metrics] * 1000, axis=0),
            )
            target_int_score = anomaly_detectors[target_node][0].score(
                int_samples[target_node].to_numpy()
            )
            flip_counts[ancestors_sorted[i]] = np.sum(
                target_int_score >= target_an_score
            )
        for i, node in enumerate(ancestors_sorted):
            if i == 0 or flip_counts[node] <= 1e-1:
                continue
            else:
                rc_scores[node] = np.log(
                    flip_counts[ancestors_sorted[i - 1]] / flip_counts[node]
                )

        sorted_rcs = sorted(rc_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            PotentialRootCause(root_cause, score) for root_cause, score in sorted_rcs
        ]

    return analyze_root_causes


# %%
