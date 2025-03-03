# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import networkx as nx
import pandas as pd

from pyrca.analyzers.rcd import RCD, RCDConfig

from pyrca.thirdparty.causallearn.utils.cit import chisq
from pyrca.thirdparty.causallearn.utils.cit import CIT
from pyrca.thirdparty.rcd import rcd

from utils.anomaly import PotentialRootCause
from utils.data_preprocessing import map_df, impute_df


def make_hierarchical_rcd(
    start_alpha: float = 0.01,
    alpha_step: float = 0.1,
    alpha_limit: float = 1,
    localized: bool = True,
    gamma: int = 5,
    bins: int = 5,
    root_cause_top_k: int = 3,
    f_node: str = "F-node",
    verbose: bool = False,
    ci_test: CIT = chisq,
    limit_metric: bool = True,
    imputation_method: str = "mean",
):
    """
    Wrapper around the RCD method as implemented in https://github.com/salesforce/PyRCA originally
        from https://github.com/azamikram/rcd.

    Paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf

    Args:
        start_alpha: The desired start significance level (float) in (0, 1) for search. Default: 0.01
        alpha_step: The search step for alpha. Default: 0.1
        alpha_limit: The maximum alpha for search. Default: 1
        localized: Whether use local method. Default: True
        gamma: Chunk size. Default: 5
        bins: The number of bins to discretize data. Default: 5
        root_cause_top_k: The maximum number of root causes in the results. Default: 3
        f_node: The name of anomaly variable. Default: "F-node"
        verbose: If verbose output should be printed. Default: False.
        ci_test: Conditional independence test. Default: chisq
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

        # remove columns for which all values are missing
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

        model = RCD(
            config=RCDConfig(
                start_alpha=start_alpha,
                alpha_step=alpha_step,
                alpha_limit=alpha_limit,
                localized=localized,
                gamma=gamma,
                bins=bins,
                k=root_cause_top_k,
                f_node=f_node,
                verbose=verbose,
                ci_test=ci_test,
            )
        )
        result = model.find_root_causes(normal_metrics, abnormal_metrics)
        potential_root_causes = []
        for idx, root_cause in enumerate(result.root_cause_nodes):
            potential_root_causes.append(
                PotentialRootCause(node=root_cause, score=1 / (idx + 1))
            )
        return potential_root_causes

    return analyze_root_causes
