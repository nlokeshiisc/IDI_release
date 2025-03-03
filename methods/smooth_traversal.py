# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List, Union, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.anomaly_scorers import ITAnomalyScorer, MeanDeviationScorer
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
from dowhy.gcm.shapley import estimate_shapley_values
import constants

warnings.filterwarnings("ignore")

gcm.config.disable_progress_bars()


def pre_process(
    *,
    causal_graph,
    target_node,
    normal_metrics,
    abnormal_metrics,
    anomaly_detection_config,
):
    """
    Builds the necessary models required for assessing the fix and anomaly condition later.
    """

    # %% 1. Train anomaly detectors for all the nodes.

    # Override the anomaly scorer with ITAnomalyScorer for Smooth Traversal.
    anomaly_detectors = {}
    for node in normal_metrics.columns:
        node_data = pd.DataFrame(normal_metrics[node]).to_numpy().reshape(-1)
        node_data = node_data[~np.isnan(node_data)]
        if np.std(node_data) == 0:
            continue
        else:
            its = ITAnomalyScorer(anomaly_scorer=MeanDeviationScorer())
            its.fit(node_data)
            anomaly_detectors[node] = its

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
    target_anomaly_scores = anomaly_detectors[target_node].score(
        abnormal_metrics[target_node].to_numpy()
    )
    trigger_step = np.argmax(target_anomaly_scores)
    logger.info(f"Trigger Step: {trigger_step}")

    abnormal_metrics = abnormal_metrics.iloc[[trigger_step]]
    target_an_score = anomaly_detectors[target_node].score(
        abnormal_metrics[target_node].to_numpy()
    )

    abnormaldf_an_scores = {
        node: anomaly_detectors[node].score(abnormal_metrics[node].to_numpy())
        for node in abnormal_metrics.columns
    }
    logger.info(
        f"Abnormal metrics raw scores\n {pd.DataFrame(abnormaldf_an_scores).to_markdown()}"
    )

    # logger.info(f"abnormal df \n{abnormal_metrics.transpose().to_markdown()}")
    logger.info(f"Target Node: {target_node} with score: {target_an_score}")
    logger.info(
        f"Target Node Mean: {normal_metrics[target_node].mean()}\t std: {normal_metrics[target_node].std()}"
    )

    return (
        normal_metrics,
        abnormal_metrics,
        anomaly_detectors,
        target_an_score,
        abnormaldf_an_scores,
    )


def make_smooth_traversal(
    n_jobs: int = -1,
    attribute_mean_deviation: bool = True,
    anomaly_scorer: gcm.anomaly_scorers.AnomalyScorer = gcm.anomaly_scorers.MeanDeviationScorer(),
    filter_for_anomalous_metrics: bool = True,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    cf_rca: bool = False,
):
    """Maker of a method to use counterfactuals for root-cause analysis.

    Args:
        n_jobs: use 1 for sequential computation, -1 for parallel.
        attribute_mean_deviation: Indicator whether the contribution is based on the feature relevance with respect
         to the given scoring function or the IT score.
        anomaly_scorer: Anomaly Scorer.
    """

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
        ) = pre_process(
            causal_graph=causal_graph,
            target_node=target_node,
            normal_metrics=normal_metrics,
            abnormal_metrics=abnormal_metrics,
            anomaly_detection_config=anomaly_detection_config,
        )

        rc_scores = {}

        ancestors = nx.ancestors(causal_graph, target_node)
        def get_rc_scores():
            """
            rc score for a node is its anomaly score minus the maximum anomaly score of its parents.
            """
            scores = {}
            for node in ancestors:
                node_raw_score = abnormaldf_an_scores[node]
                parents = get_ordered_predecessors(causal_graph, node)
                parent_scores = []
                for p in parents:
                    parent_scores.append(abnormaldf_an_scores[p])
                max_parent_score = max(parent_scores) if len(parent_scores) > 0 else 0
                scores[node] = node_raw_score - max_parent_score
            return scores

        rc_scores = get_rc_scores()
        
        sorted_rcs = sorted(rc_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            PotentialRootCause(root_cause, score) for root_cause, score in sorted_rcs
        ]

    return analyze_root_causes
