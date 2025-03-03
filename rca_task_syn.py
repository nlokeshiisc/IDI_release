from constants import logger
import constants
from utils.anomaly import PotentialRootCause
import networkx as nx
import pandas as pd
import numpy as np
from typing import List
import random
from utils.simulation_utils import create_syn_testcase
from dowhy.gcm import InvertibleStructuralCausalModel
from utils.plot_utils import plot_nx_graph
from pathlib import Path
import os


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)


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


def evaluate_syn(
    method_name,
    method_function,
    num_test_cases: int = 10,
    num_root_causes: int = 1,
    one_cause_per_path: bool = True,
    linear_eqns: bool = True,
    invertible: bool = True,
) -> pd.DataFrame:
    """Computes the top-k recall of the method to analyze root causes.

    Args:
        analyze_root_causes: Method that produces potential root causes for a given SLO violation.
        dir: Directory of the benchmarking dataset.
        split: Split of train or test, defaults to None using both.

    Returns: A DataFrame with the top-1 and top-3 recall.
    """

    results = {}
    result_list = []
    dataset_name = "syn_linear" if linear_eqns else "syn_nonlinear"
    constants.RESULTS_DIR = (
        Path("results")
        / dataset_name
        / f"rc-{num_root_causes}"
        / f"ocpp-{one_cause_per_path}"
    )
    if invertible == False:
        constants.RESULTS_DIR = constants.RESULTS_DIR / "non-invertible"
    constants.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (constants.RESULTS_DIR / "graph").mkdir(parents=True, exist_ok=True)
    (constants.RESULTS_DIR / "data").mkdir(parents=True, exist_ok=True)
    (constants.RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    file_name = f"{method_name}.log"
    constants.set_logger(dir=constants.RESULTS_DIR / "logs", file_name=file_name)

    if num_root_causes == 1:
        constants.UNQ_RC_EXPT = True
    else:
        constants.UNQ_RC_EXPT = False

    for tc_idx in range(num_test_cases):
        # set seed here for reproducibility
        set_seed(tc_idx * 100)
        logger.info("****" * 10 + f"Test Case {tc_idx}" + "****" * 10)

        data_dict = create_syn_testcase(
            num_root_causes=num_root_causes,
            num_normal_data=constants.NORMAL_DATA_SIZE,
            one_cause_per_path=one_cause_per_path,
            linear_eqns=linear_eqns,
            invertible=invertible,
        )
        normal_metrics: pd.DataFrame = data_dict["normal_metrics"]
        abnormal_metrics: pd.DataFrame = data_dict["abnormal_metrics"]

        normal_means = normal_metrics.mean()
        normal_stds = normal_metrics.std()
        normal_metrics = (normal_metrics - normal_means) / normal_stds
        abnormal_metrics = (abnormal_metrics - normal_means) / normal_stds

        if not os.path.exists(constants.RESULTS_DIR / "data" / f"{tc_idx}-normal.csv"):
            normal_metrics.to_csv(
                constants.RESULTS_DIR / "data" / f"{tc_idx}-normal.csv"
            )
            abnormal_metrics.to_csv(
                constants.RESULTS_DIR / "data" / f"{tc_idx}-abnormal.csv"
            )

        target_node = data_dict["target_node"]
        root_cause_nodes = data_dict["root_cause_nodes"]
        scm: InvertibleStructuralCausalModel = data_dict["scm"]

        results[tc_idx] = {1: [], 3: []}
        assert isinstance(root_cause_nodes, list), "Pass the root cause nodes as a list"
        assert (
            abnormal_metrics.shape[0] == 1
        ), "pass one abnormal entry for each test case"
        # This is to be consistent with petshop code that has been hard coded to work with 5 instances.
        abnormal_metrics = pd.concat([abnormal_metrics] * 5)

        # The above kwargs are used only to debug the int_rca and cf_rca methods. They are not used explicitly inside the methods.
        potential_root_causes = method_function(
            causal_graph=scm.graph,
            target_node=target_node,
            normal_metrics=normal_metrics,
            abnormal_metrics=abnormal_metrics,
        )
        potential_root_causes = sorted(
            potential_root_causes, key=lambda x: x.score, reverse=True
        )
        predicted_causes = [entry.node for entry in potential_root_causes]
        num_gt_rcs = len(root_cause_nodes)
        for k in [num_gt_rcs, num_gt_rcs + 2]:  # reminiscent of Recall@1, 3
            recall = recall_at_k(
                predicted_causes=predicted_causes[:k],
                ground_truth_causes=root_cause_nodes,
                k=k,
            )
            row = {
                "num_causes": num_root_causes,
                "one_cause_per_path": one_cause_per_path,
                "k": k - num_gt_rcs + 1,
                "target_node": target_node,
                "gt_root_causes": root_cause_nodes,
                "predicted_causes": "::".join(predicted_causes),
                "Recall": recall,
            }
            result_list.append(row)
    return pd.DataFrame(result_list)


def recall_at_k(predicted_causes, ground_truth_causes, k):
    """Computes the recall for a given k

    Args:
        potential_root_causes (_type_): _description_
        ground_truth_causes (_type_): _description_
        k (_type_): _description_
    """
    assert k >= len(ground_truth_causes)
    int_nodes = set(ground_truth_causes).intersection(set(predicted_causes[:k]))
    return len(int_nodes) / len(ground_truth_causes)
