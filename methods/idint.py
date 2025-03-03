# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List, Union, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.whatif import interventional_samples, counterfactual_samples
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
import torch

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

    # %% 1. Train anomaly detectors for all the nodes.
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
    logger.info(f"Trigger Step: {trigger_step}")

    abnormal_metrics = abnormal_metrics.iloc[[trigger_step]]
    target_an_score = anomaly_detectors[target_node][0].score(
        abnormal_metrics[target_node].to_numpy()
    )

    abnormaldf_an_scores = {
        node: anomaly_detectors[node][0].score(abnormal_metrics[node].to_numpy())
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

    # %% 5. Fit an SCM for performing interventions.
    # Train a structural causal model on normal data for the target_metric and average statistic.
    learned_scm: gcm.ProbabilisticCausalModel = train_scm(
        causal_graph.copy(),
        normal_metrics,
        linear_eqns=linear_eqns,
    )
    # evaluate_scm(scm, normal_metrics)

    return (
        normal_metrics,
        abnormal_metrics,
        anomaly_detectors,
        target_an_score,
        abnormaldf_an_scores,
        learned_scm,
    )


def make_idint(
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

        rc_scores = {}

        ancestors = nx.ancestors(causal_graph, target_node)

        def assess_anomaly_condition():
            """This is the anomaly condition for the potential root cause nodes.
            A node is a potential root cause if the following holds:
                1. The node itself is anomalous
                2. The parents are not anomalous
            Rerurns:
                List of potential root causes.
            """
            potential_rcs = []
            for node in ancestors:
                node_raw_score = abnormaldf_an_scores[node]

                parents = get_ordered_predecessors(causal_graph, node)
                anomalous_parent = None
                for p in parents:
                    if abnormaldf_an_scores[p] > p_an_thresh:
                        anomalous_parent = p
                        break

                if anomalous_parent is not None:
                    pass
                else:
                    potential_rcs.append(node)

            return potential_rcs

        potential_rcs = assess_anomaly_condition()
        rcs_idx, rcs_inv_idx = {}, {}
        for i, rc in enumerate(potential_rcs):
            rcs_idx[rc] = i
            rcs_inv_idx[i] = rc

        logger.info(f"Potential Root Causes after anomaly condition: {potential_rcs}")

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

        int_nodes = _est_fix_nodes(potential_rcs)

        def _est_uniq_rc_scores():
            rc_scores = {}
            for rc in potential_rcs:
                int_dict = {rc: lambda node: int_nodes[rc]}
                if cf_rca == False:
                    int_samples = interventional_samples(
                        causal_model=learned_scm,
                        interventions=int_dict,
                        observed_data=abnormal_metrics,
                    )
                else:
                    int_samples = counterfactual_samples(
                        causal_model=learned_scm,
                        interventions=int_dict,
                        observed_data=abnormal_metrics,
                    )
                target_int_score = anomaly_detectors[target_node][0].score(
                    int_samples[target_node].to_numpy()
                )
                if target_int_score < target_an_score - 1e-3:
                    rc_scores[rc] = target_an_score - target_int_score
            return rc_scores

        @torch.no_grad()
        def set_function(alpha: np.ndarray) -> Union[np.ndarray, float]:
            int_dict = {}
            for i, s in enumerate(alpha):
                if s == 1:
                    int_dict[rcs_inv_idx[i]] = lambda x: int_nodes[rcs_inv_idx[i]]

            int_samples = abnormal_metrics.copy()
            if len(int_dict) > 0:
                if cf_rca == False:
                    int_samples = interventional_samples(
                        causal_model=learned_scm,
                        interventions=int_dict,
                        observed_data=abnormal_metrics,
                    )
                else:
                    int_samples = counterfactual_samples(
                        causal_model=learned_scm,
                        interventions=int_dict,
                        observed_data=abnormal_metrics,
                    )
            target_int_score = anomaly_detectors[target_node][0].score(
                int_samples[target_node].to_numpy()
            )
            return np.array([target_an_score - target_int_score])

        if constants.UNQ_RC_EXPT == False:
            shap_config = constants.SHAPLEY_CONFIG_APPROX
            if len(potential_rcs) <= 5:
                shap_config = constants.SHAPLEY_CONFIG_EXACT

            rc_scores = estimate_shapley_values(
                set_func=set_function,
                num_players=len(potential_rcs),
                shapley_config=shap_config,
            )
            rc_scores = np.squeeze(rc_scores).reshape(-1)
            # TODO: Discard nodes with negative shapley values.
            rc_scores = {
                rcs_inv_idx[idx]: score
                for idx, score in enumerate(rc_scores)  # if score > 1e-8
            }
        else:
            rc_scores = _est_uniq_rc_scores()

        sorted_rcs = sorted(rc_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            PotentialRootCause(root_cause, score) for root_cause, score in sorted_rcs
        ]

    return analyze_root_causes
