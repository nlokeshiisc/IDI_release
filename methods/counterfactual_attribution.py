# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List, Union, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.auto import AssignmentQuality
from utils.data_preprocessing import pad_and_replace_nan
from utils.scm_utils import train_scm

from utils.anomaly import PotentialRootCause
from utils.data_preprocessing import reduce_df, pad_and_fill
import constants


warnings.filterwarnings("ignore")

gcm.config.disable_progress_bars()


def make_counterfactual_attribution_method(
    n_jobs: int = -1,
    attribute_mean_deviation: bool = False,
    anomaly_scorer: gcm.anomaly_scorers.AnomalyScorer = gcm.anomaly_scorers.MeanDeviationScorer(),
    linear_eqns: bool = True,
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
        """Method to identify potential root causes of a performance issue in target_node through counterfactuals.

        Implementation of Budhathoki et al. "Causal structure based root cause analysis of outliers" in ICML '22.
        https://arxiv.org/abs/1912.02724

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violoation to investigate.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """

        normal_df = pad_and_replace_nan(
            normal_metrics.copy(), required_columns=causal_graph.nodes
        )
        abnormal_df, original_abnormal_columns = pad_and_fill(
            abnormal_metrics.copy(), fill_df=normal_df
        )
        abnormal_df = abnormal_df.iloc[2:3]
        # Train a structural causal model on normal data for the target_metric and average statistic.
        scm = train_scm(
            causal_graph,
            normal_df,
            linear_eqns=linear_eqns,
        )

        scores = gcm.attribute_anomalies(
            scm,
            target_node=target_node,
            anomaly_samples=abnormal_df,
            shapley_config=constants.SHAPLEY_CONFIG_APPROX,  # Exact takes a lot of time
            attribute_mean_deviation=attribute_mean_deviation,
            anomaly_scorer=anomaly_scorer,
        )
        # Filter out scores for columns for which we had no measurement in abnormal.
        for c in scores:
            if c not in original_abnormal_columns:
                scores[c] = np.zeros(len(scores[c]))
            scores[c][np.isnan(scores[c])] = 0

        return [
            PotentialRootCause(root_cause, np.mean(scores))
            for (root_cause, scores) in scores.items()
        ]

    return analyze_root_causes
