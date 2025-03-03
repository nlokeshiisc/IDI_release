import copy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings
from utils.data_preprocessing import reduce_df
from typing import Dict, List, Any, Tuple, Union, Callable, Optional, Set
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from constants import AnomalyDetectionConfig, DEFAULT_ANOMALY_DETECTION


@dataclass
class PotentialRootCause:
    """Class representing one potential root cause.

    Attributes:
        node: Node in the microservice architecture that could have caused a performance issue in the application.
        metric: Metric within that node that describes the cause of the performance issue.
        score: A score capturing the likelihood that this node is the true root cause.
    """

    node: str
    # metric: str # Always use the target metric from the ground_truth metric only for evaluation.
    score: float


# === ANOMALY DETECTION === #


class MeanDeviationWithoutRescaling(AnomalyScorer):
    """The score here is simply the deviation from the mean."""

    def __init__(self, mean: Optional[float] = None):
        self._mean = mean

    def fit(self, X: np.ndarray) -> None:
        if self._mean is None:
            self._mean = np.mean(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        return abs(X.reshape(-1) - self._mean)


def train_anomaly_detectors(
    metrics,
    anomaly_detection_config: AnomalyDetectionConfig,
) -> Dict[str, Tuple[AnomalyScorer, AnomalyDetectionConfig]]:
    """Trains an anomaly detector for each service defined by the anomaly detection config. The fitted models are
    stored as dictionary.

    :param training_end_time: The end time stamp for the training period.
    :param lookback_period: The lookback period for the training period.
    :param aggregation_period: The aggregation period for the training period.
    :param anomaly_detection_config: The configuration for the anomaly detection.
    :param metric: The metric to use for training.
    :param stat: The statistic to use for training.
    :param filename: The filename to store the trained anomaly detectors.
    :return: A dictionary mapping service names to the trained anomaly detectors with their config (tuple).
    """
    trained_ads = {}

    data_matrix = metrics.copy()
    for c in data_matrix.columns:
        training_data = pd.DataFrame(data_matrix[c]).to_numpy().reshape(-1)
        training_data = training_data[~np.isnan(training_data)]
        if training_data.shape[0] < 10:
            warnings.warn(
                "After removing missing data, %s has fewer than 10 data points! Using no model instead."
                % c
            )
            continue
        elif np.std(training_data) == 0:
            warnings.warn(
                "The standard deviation of %s is 0. Using a trivial model instead." % c
            )
            scorer = MeanDeviationWithoutRescaling()
            scorer.fit(training_data)
            tmp_config = AnomalyDetectionConfig(
                MeanDeviationWithoutRescaling,
                False,
                0,
                anomaly_detection_config.description,
                description="ZScoreNoScale",
            )
        else:
            scorer = anomaly_detection_config.anomaly_scorer()
            scorer.fit(training_data)
            tmp_config = copy.deepcopy(anomaly_detection_config)

        trained_ads[c] = (scorer, tmp_config)

    return trained_ads


def estimate_anomaly_scores(
    data: pd.DataFrame,
    anomaly_scorer: AnomalyScorer,
    is_it_score: bool,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate anomaly scores based on the given data.

    :param data: A pandas dataframe containing the data to be scored.
    :param anomaly_scorer: An instance of AnomalyScorer to use for calculating the anomaly scores.
    :param is_it_score: A boolean indicating whether the scores returned by the anomaly scorer should be treated as IT
                        scores. This is, they would be converted to p-values based on exp(-log_prob)
    :param threshold: A float value indicating the threshold above which a score will be considered anomalous.
    :return: A tuple containing two numpy arrays, where the first entry is a binary decision whether a point is
             anomalous and the second entry is the corresponding score.
    """
    data = np.array(data.to_numpy()).reshape(-1)
    non_nan_values = data[~np.isnan(data)]
    scores = np.zeros(data.shape[0])
    if non_nan_values.shape[0] == 0:
        return np.array([False] * data.shape[0]), np.zeros(data.shape[0])

    tmp_scores = anomaly_scorer.score(non_nan_values).reshape(-1)
    if is_it_score:
        tmp_scores = 1 - np.exp(-tmp_scores)
    scores[~np.isnan(data)] = tmp_scores
    scores[np.isnan(scores)] = 0

    return scores > threshold, scores


def estimate_binary_decision_and_anomaly_scores(
    data_matrix: pd.DataFrame,
    anomaly_detectors: Dict[str, Tuple[AnomalyScorer, AnomalyDetectionConfig]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Computes the binary decision and anomaly scores.

    :param data_matrix: The data matrix to score.
    :param anomaly_detectors: The anomaly detectors.
    :return: Two dictionaries, where the first one is the binary decision and the second one the anomaly scores for
             each service.
    """
    binary_anomaly_indicators = {}
    anomaly_scores = {}

    for c in data_matrix.columns:
        if c not in anomaly_detectors or anomaly_detectors[c] is None:
            warnings.warn(
                "WARNING: No anomaly scorer found for %s! Will skip this metric and assume there "
                "are no anomalies." % c
            )
            tmp_indicator = np.array([False] * data_matrix.shape[0])
            tmp_scores = np.zeros(data_matrix.shape[0])
        else:
            tmp_indicator, tmp_scores = estimate_anomaly_scores(
                data_matrix[c],
                anomaly_detectors[c][0],
                anomaly_detectors[c][1].convert_to_p_value,
                anomaly_detectors[c][1].anomaly_score_threshold,
            )
        binary_anomaly_indicators[c] = tmp_indicator
        anomaly_scores[c] = tmp_scores

    return binary_anomaly_indicators, anomaly_scores


def get_anomalous_metrics_and_scores(
    normal_metrics: pd.DataFrame,
    abnormal_metrics: pd.DataFrame,
    target_node: str,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    search_for_anomaly: bool = True,
):
    """
    Trains anomaly detectors for each microservice in normal metrics.
    Then for each timestep in abnomral metrics, it computes the anomaly score for each microservice.
    """
    # 1. train anomaly detectors
    anomaly_detectors = train_anomaly_detectors(
        normal_metrics,
        anomaly_detection_config,
    )
    data_matrix = abnormal_metrics.copy()

    # 2.a) Get binary indicator for each timestamp whether it was anomalous for a specific node.
    (
        binary_anomaly_indicators,
        anomaly_scores,
    ) = estimate_binary_decision_and_anomaly_scores(data_matrix, anomaly_detectors)

    # 2.b) Adjust trigger point. This is needed if the given trigger point does not exactly coincide with the indices of the
    # data matrix. Here, we just pick the third point in time to give the issues some time to trigger.

    row_index_to_analyze = min(2, data_matrix.shape[0] - 1)
    initial_row_index = row_index_to_analyze

    if search_for_anomaly:
        for i in range(row_index_to_analyze, data_matrix.shape[0]):
            if binary_anomaly_indicators[target_node][i]:
                row_index_to_analyze = i
                break
    if row_index_to_analyze != initial_row_index:
        warnings.warn(
            "WARNING: The given trigger point %s was not anomalous. Using point %s instead, since this is the next "
            "timestamp that has been identified as anomalous."
            % (data_matrix.index[2], data_matrix.index[row_index_to_analyze])
        )

    if not binary_anomaly_indicators[target_node][row_index_to_analyze]:
        warnings.warn(
            "Target was not considered anomalous by the anomaly detector for the given trigger point!"
        )
        return {}, []

    return {
        c: anomaly_scores[c][row_index_to_analyze]
        for c in data_matrix.columns
        if binary_anomaly_indicators[c][row_index_to_analyze]
    }
