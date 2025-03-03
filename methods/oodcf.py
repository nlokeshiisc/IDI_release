# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dowhy import gcm

gcm.config.disable_progress_bars()
from methods import idint
from constants import (
    AnomalyDetectionConfig,
    DEFAULT_ANOMALY_DETECTION,
)


def make_ood_cf(
    n_jobs: int = -1,
    attribute_mean_deviation: bool = True,
    anomaly_scorer: gcm.anomaly_scorers.AnomalyScorer = gcm.anomaly_scorers.MeanDeviationScorer(),
    filter_for_anomalous_metrics: bool = True,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    cf_rca: bool = False,
    linear_eqns: bool = True,
):
    return idint.make_idint(
        n_jobs=n_jobs,
        attribute_mean_deviation=attribute_mean_deviation,
        anomaly_scorer=anomaly_scorer,
        filter_for_anomalous_metrics=filter_for_anomalous_metrics,
        anomaly_detection_config=anomaly_detection_config,
        cf_rca=True,
        linear_eqns=linear_eqns,
    )
