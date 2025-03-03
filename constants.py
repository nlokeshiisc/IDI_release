from logging import Logger
import logging
from dowhy.gcm.anomaly_scorers import MeanDeviationScorer
from dataclasses import dataclass
from typing import Callable
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from pathlib import Path
from dataclasses import replace
from dowhy.gcm.shapley import ShapleyConfig, ShapleyApproximationMethods
import numpy as np


@dataclass
class AnomalyDetectionConfig:
    """This class represents the configuration for an anomaly detector.

    Attributes:
        anomaly_scorer: A callable that produces an `AnomalyScorer` object.
        convert_to_p_value: A bool indicating whether to convert the scores to p-values. This only makes sense if the
                            anomaly_scorer returns scores. This does np.exp(-score).
        anomaly_score_threshold: The threshold used to determine whether a score is anomalous.
        description: A description of the configuration."""

    anomaly_scorer: Callable[[], AnomalyScorer]
    convert_to_p_value: bool
    anomaly_score_threshold: float
    parent_anomaly_score_thresold: float
    description: str


DEFAULT_ANOMALY_DETECTION = AnomalyDetectionConfig(
    anomaly_scorer=MeanDeviationScorer,
    convert_to_p_value=False,
    anomaly_score_threshold=5,
    parent_anomaly_score_thresold=5,
    description="MeanDScore",
)

anomaly_configs = {
    "petshop": {
        "anomaly_score_threshold": 5,
        "parent_anomaly_score_thresold": 5,
    },  # Borrowed from Petshop
    "syn_linear": {
        "anomaly_score_threshold": 2, # Follow 2 \sigma rule; do not tune!  
        "parent_anomaly_score_thresold": 2,
    },
    "syn_nonlinear": {
        "anomaly_score_threshold": 2,
        "parent_anomaly_score_thresold": 2,
    },
}


def update_anomaly_config(dataset_name: str, **kwargs):
    global DEFAULT_ANOMALY_DETECTION, anomaly_configs

    if dataset_name == "syn":
        dataset_name = f"syn_linear" if kwargs["linear_eqns"] else "syn_nonlinear"
    DEFAULT_ANOMALY_DETECTION = replace(
        DEFAULT_ANOMALY_DETECTION, **anomaly_configs[dataset_name]
    )


RESULTS_DIR = Path("results")
DEVICE = "cuda:2"


# gold = "#E6BE8A"
COLOR_RC = "gold"  # "purple"
COLOR_NRM = "lightblue"  # "lightblue"
OPACITY = 0.5
COLOR_TGT = "purple"  # "salmon"  # "lightsalmon"
COLOR_RCPRED = "purple"
COLOR_WEAK = "cyan"

SHAPLEY_CONFIG_APPROX = ShapleyConfig(
    approximation_method=ShapleyApproximationMethods.PERMUTATION,
    num_samples=500,
    n_jobs=10,
)

SHAPLEY_CONFIG_EXACT = ShapleyConfig(
    approximation_method=ShapleyApproximationMethods.EXACT,
    n_jobs=10,
)

NORMAL_DATA_SIZE: int = 50
UNQ_RC_EXPT: bool = True

# %% Noise parameters
WEAK_FACTOR: float = 0.05
__unf_a = 0
__unf_b = 1
__norm_mu = 0
__norm_sigma = 1
EXOG_DIST = "uniform"
NOISE_DIST = {
    "exog": {
        "uniform": {"a": __unf_a, "b": __unf_b},
        "norm": {"mu": __norm_mu, "sigma": __norm_sigma},
    },
    "lin_model": {"a": 0.5, "b": 2},
    "mlp_model": {"a": 0, "b": 1},
}
exog_mean_fn = lambda: np.array(
    [(__unf_a + __unf_b) / 2 if EXOG_DIST == "uniform" else __norm_mu]
)
WEAK_NODE_NORMAL_MECHANISM = False
# %% End Noise parameters


def set_logger(dir: Path = None, file_name: str = "experiment.log"):
    global logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_dir = dir if dir is not None else Path("results")
    logging.basicConfig(
        filename=log_dir / file_name,
        format="%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        filemode="w",
    )
    logger = logging.getLogger(name="shapley_expts")
    logger.setLevel(logging.DEBUG)


logger: Logger = None
try:
    set_logger()
except Exception as e:
    pass

# Some named keys in dowhy
# This constant is used as key when storing/accessing models as causal mechanisms in graph node attributes
CAUSAL_MECHANISM = "causal_mechanism"

# This constant is used as key when storing the parents of a node during fitting. It's used for validation purposes afterwards.
PARENTS_DURING_FIT = "parents_during_fit"
