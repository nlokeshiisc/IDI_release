from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.graph import get_ordered_predecessors
from dowhy.gcm.ml.regression import SklearnRegressionModel
import constants
from dowhy.gcm.graph import is_root_node
import numpy as np
import constants
from typing import Tuple
from dowhy.gcm import AdditiveNoiseModel
import constants
import random
import networkx as nx
from utils.model_utils import MyMLPModel, MyLinearRegressionModel, MyNonInvMLPModel
import torch
from scipy import stats

from dowhy.gcm import (
    InvertibleStructuralCausalModel,
    ScipyDistribution,
    AdditiveNoiseModel,
)


def get_exogenous_dist(weak: bool = False):
    weak_factor = constants.WEAK_FACTOR if weak else 1
    if constants.EXOG_DIST == "uniform":
        a = constants.NOISE_DIST["exog"]["uniform"]["a"]
        b = constants.NOISE_DIST["exog"]["uniform"]["b"] * weak_factor
        return ScipyDistribution(stats.uniform, loc=a, scale=b)
    elif constants.EXOG_DIST == "norm":
        mu = constants.NOISE_DIST["exog"]["norm"]["mu"]
        sigma = constants.NOISE_DIST["exog"]["norm"]["sigma"] * weak_factor
        return ScipyDistribution(stats.norm, loc=mu, scale=sigma)
    else:
        raise ValueError(f"Unknown Exogenous noise distribution {constants.EXOG_DIST}")


def sample_exogenous_vars(weak, num_vars: int) -> np.array:
    weak_factor = constants.WEAK_FACTOR if weak else 1
    if constants.EXOG_DIST == "uniform":
        a = constants.NOISE_DIST["exog"]["uniform"]["a"]
        b = constants.NOISE_DIST["exog"]["uniform"]["b"] * weak_factor
        return np.random.uniform(a, b, num_vars)
    elif constants.EXOG_DIST == "norm":
        mu = constants.NOISE_DIST["exog"]["norm"]["mu"]
        sigma = constants.NOISE_DIST["exog"]["norm"]["sigma"] * weak_factor
        return np.random.normal(mu, sigma, num_vars)
    else:
        raise ValueError(f"Unknown Exogenous noise distribution {constants.EXOG_DIST}")


def sample_natural_number(init_mass) -> int:
    current_mass = init_mass
    probability = np.random.uniform(0, 1)
    k = 1
    is_searching = True
    while is_searching:
        if probability <= current_mass:
            return k
        else:
            k += 1
            current_mass += 1 / (k**2)


def random_scm(
    num_root_nodes, num_internal_nodes, linear_eqns, invertible
) -> InvertibleStructuralCausalModel:
    """Samples a random SCM
    Total nodes = num_root_nodes + num_internal_nodes
    Assigns the causal mechanism at each node as Linear Regression if linear_eqns is True; else MLP
    """
    scm = InvertibleStructuralCausalModel(nx.DiGraph())
    causal_graph: nx.DiGraph = scm.graph
    all_nodes = []

    for i in range(num_root_nodes):
        new_root = "X" + str(i)
        causal_graph.add_node(new_root)
        scm.set_causal_mechanism(new_root, get_exogenous_dist())
        causal_graph.nodes[new_root][
            constants.PARENTS_DURING_FIT
        ] = get_ordered_predecessors(scm.graph, new_root)
        all_nodes.append(new_root)

    for i in range(num_internal_nodes):
        parents = np.random.choice(
            all_nodes,
            min(sample_natural_number(init_mass=0.6), len(all_nodes)),
            replace=False,
        )

        new_child = "X" + str(i + num_root_nodes)
        causal_graph.add_node(new_child)

        # Assign the causal mechanism to new_child
        assign_mechanism(
            node=new_child,
            parents=parents,
            linear_eqns=linear_eqns,
            scm=scm,
            weak=False,
            invertible=invertible,
        )

        for parent in parents:
            causal_graph.add_edge(parent, new_child)

        causal_graph.nodes[new_child][
            constants.PARENTS_DURING_FIT
        ] = get_ordered_predecessors(scm.graph, new_child)
        all_nodes.append(new_child)
    return scm


def assign_mechanism(
    *,
    node,
    parents,
    linear_eqns,
    scm: InvertibleStructuralCausalModel,
    weak: bool,
    invertible: bool,
):
    """Assigns the causal mechanism to the node based on the parents"""
    if linear_eqns == True:
        model = MyLinearRegressionModel(num_parents=len(parents), node=node)
    else:
        if invertible == True:
            model = MyMLPModel(
                len(parents),
                node=node,
                num_hidden=4,
                num_layers=1,
                act_fn=torch.nn.ELU,
                use_layer_norm=False,
            )
        else:
            model = MyNonInvMLPModel(
                len(parents),
                node=node,
                num_hidden=4,
                num_layers=1,
                act_fn=torch.nn.ELU,
                use_layer_norm=False,
                exog_dist_fn=lambda num_samples: sample_exogenous_vars(
                    weak=weak, num_vars=num_samples
                ),
            )
        model.to(constants.DEVICE)
        model.set_grads_enabled(False)

    model.init_parameters(weak=weak)
    if invertible == True:
        causal_mechanism = AdditiveNoiseModel(model, get_exogenous_dist(weak))
    else:
        causal_mechanism = AdditiveNoiseModel(
            model, ScipyDistribution(stats.uniform, loc=0, scale=0.005)
        )  # Noise is there in the MLP itself
    scm.set_causal_mechanism(node, causal_mechanism)


def create_scm(
    linear_eqns: bool, invertible: bool
) -> Tuple[InvertibleStructuralCausalModel, str]:
    """
    Creates a DAG with the following properties:
        1. Number of nodes: 20-30
        2. Number of root nodes: 1-5
        3. Target node: A node with at least 10 ancestors.
        4. Causal mechanisms: Linear regression models.
        5. Misspecified causal mechanism: If True, the causal mechanism is purposely misspecified to a squared regression model.
        6. Learns a causal model from the data.
    Returns:
        ground_truth_dag: The ground truth DAG.
        target_node: The target node.
        learned_dag: The learned DAG.
    """

    is_sufficiently_deep_graph = False
    while not is_sufficiently_deep_graph:
        # Generate DAG with random number of nodes and root nodes
        num_total_nodes = np.random.randint(20, 30)
        num_root_nodes = np.random.randint(1, 5)

        ground_truth_scm = random_scm(
            num_root_nodes,
            num_internal_nodes=num_total_nodes - num_root_nodes,
            linear_eqns=linear_eqns,
            invertible=invertible,
        )
        # Choose the node with the most ancestors as the target node
        target_node = max(
            ground_truth_scm.graph.nodes(),
            key=lambda node: len(nx.ancestors(ground_truth_scm.graph, node)),
        )
        if len(nx.ancestors(ground_truth_scm.graph, target_node)) >= 10:
            constants.logger.info(
                f"Target node: {target_node}; Ancestors: {len(nx.ancestors(ground_truth_scm.graph, target_node))}"
            )
            break
    return ground_truth_scm, target_node


def evaluate_scm(scm: gcm.InvertibleStructuralCausalModel, metrics: pd.DataFrame):
    """
    Evaluates the performance of the trained SCM on the given metrics dataframe.
    This gives us an idea of how good the DGP is fit to the data.
    Training error being more is indicative of poor model choices and hence we need to consider heavy models like neural networks, GDBT, etc.
    """
    for node in scm.graph.nodes():
        parents = get_ordered_predecessors(scm.graph, node)
        if len(parents) == 0:
            continue
        node_mechanism = scm.causal_mechanism(node)
        x = metrics[parents].to_numpy()
        y = metrics[node].to_numpy()
        y_hat = node_mechanism.draw_samples(x)
        error = ((np.reshape(y, -1) - np.reshape(y_hat, -1)) ** 2).mean()
        if error > 1e-1:
            constants.logger.info(f"Node {node} has an error of {error}")
    return


def train_scm(
    causal_graph: nx.DiGraph,
    normal_metrics: pd.DataFrame,
    linear_eqns: bool = False,
) -> Tuple[gcm.InvertibleStructuralCausalModel, pd.DataFrame]:
    scm = gcm.InvertibleStructuralCausalModel(causal_graph)

    if linear_eqns:
        for node in causal_graph.nodes:
            parents = list(causal_graph.predecessors(node))
            if len(parents) == 0:
                scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
            else:
                scm.set_causal_mechanism(
                    node, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
                )
    else:
        for node in causal_graph.nodes:
            parents = list(causal_graph.predecessors(node))
            if len(parents) == 0:
                scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
            else:
                # MLP based causal mechanisms
                mlp_model = MyMLPModel(
                    num_parents=len(parents),
                    node=node,
                    num_hidden=8,
                    num_layers=2,
                    act_fn=torch.nn.ELU,
                    use_layer_norm=False,
                )
                mlp_model.to(constants.DEVICE)
                scm.set_causal_mechanism(
                    node,
                    gcm.AdditiveNoiseModel(mlp_model),
                )
    gcm.fit(scm, normal_metrics)
    return scm


def generate_x_from_noise(scm, exogenous_vars) -> dict:
    causal_graph = scm.graph
    obs_vars = {}
    for i, node in enumerate(nx.topological_sort(scm.graph)):
        node_mechanism = scm.causal_mechanism(node)

        if is_root_node(scm.graph, node):
            obs_vars[node] = exogenous_vars[node]
        else:
            parent_ids = get_ordered_predecessors(causal_graph, node)
            parent_samples = np.column_stack([obs_vars[i] for i in parent_ids])
            obs_vars[node] = node_mechanism.evaluate(
                parent_samples, exogenous_vars[node]
            ).reshape(-1)
    return obs_vars
