import numpy as np
import constants
from dowhy.gcm import draw_samples
import constants
import random
import networkx as nx
import utils.common_utils as cu
import random

from dowhy.gcm import InvertibleStructuralCausalModel, AdditiveNoiseModel
from dowhy.gcm.graph import get_ordered_predecessors
import pandas as pd
import constants
import networkx as nx
import utils.scm_utils as su


def get_node_coeffs(linear_eqns, scm, target_node) -> dict:
    assert linear_eqns == True, "This function is only valid for linear models"
    gt_node_coeffs = {}
    causal_graph = scm.graph
    for node in causal_graph.nodes:
        all_paths = list(nx.all_simple_paths(causal_graph, node, target_node))
        if len(all_paths) == 0:
            gt_node_coeffs[node] = [0]
            continue
        node_coeff = 0
        for path in all_paths:
            path.reverse()
            tmp_coef = 1
            for i in range(0, len(path) - 1):
                current_node = path[i]
                upstream_node = path[i + 1]
                parent_coef_index = get_ordered_predecessors(
                    scm.graph, current_node
                ).index(upstream_node)
                tmp_coef *= scm.causal_mechanism(
                    current_node
                ).prediction_model.sklearn_model.coef_[parent_coef_index]
            node_coeff += tmp_coef
        gt_node_coeffs[node] = [node_coeff]
    gt_node_coeffs[target_node] = [1]
    return gt_node_coeffs


def draw_root_causes(
    *,
    scm: InvertibleStructuralCausalModel,
    num_causes: int,
    target_node: str,
    one_cause_per_path=False,
) -> list:
    """Draws anomaly data from the SCM"""
    root_causes = []

    causal_graph: nx.DiGraph = scm.graph
    target_ancestors = sorted(list(nx.ancestors(causal_graph, target_node)))

    # This is to ensure that every simple path has atmost one root cause
    if one_cause_per_path == False:
        root_causes = random.sample(target_ancestors, num_causes)
    else:
        paths = cu.simple_paths_to_target(causal_graph, target_node, target_ancestors)
        candidate_causes = target_ancestors.copy()

        # fmt: off
        for i in range(num_causes):
            cause = np.random.choice(candidate_causes)
            [[candidate_causes.remove(n) for n in p if n in candidate_causes] for p in paths if cause in p]
            root_causes.append(cause)
            if len(candidate_causes) == 0:
                break
        # fmt: on

    return root_causes


def __get_rc_noise(
    scm: InvertibleStructuralCausalModel,
    target_node: str,
    linear_eqns: bool,
    root_causes: list,
) -> dict:
    """This is the search for a sample from Q^{RC(j)}_\epsilon such that nodes in root_causes are root causes for anomaly at target_node
    Args:
        scm (_type_): _description_
        target_node (_type_): _description_
        linear_eqns (_type_): _description_
        anomaly_noise (_type_): _description_
        root_causes (_type_): _description_
        causal_graph (_type_): _description_
    """

    anomaly_noise = {node: constants.exog_mean_fn() for node in scm.graph.nodes}
    normal_metrics = draw_samples(scm, num_samples=1000)
    node_means = normal_metrics.mean()
    node_stds = normal_metrics.std()
    node_score_fn = lambda node, x: abs(x - node_means[node]) / node_stds[node]

    rc_eps = constants.exog_mean_fn()

    # Ensure that the root cause node induce anomaly at the target node
    while True:
        rc_eps += 0.5
        for rc in root_causes:
            anomaly_noise[rc] = rc_eps
        anomaly_data = su.generate_x_from_noise(scm, anomaly_noise)
        target_score = node_score_fn(target_node, anomaly_data[target_node][0])
        if target_score >= 5:
            break

    constants.logger.info(
        f"rc_eps at root cause search: {rc_eps}", extra={"tags": "scm"}
    )

    # Ensure that the root cause nodes are anomalous
    for rc in root_causes:
        flag = False
        while flag == False:
            anomaly_data = su.generate_x_from_noise(scm, anomaly_noise)
            rc_score = node_score_fn(rc, anomaly_data[rc][0])
            if rc_score >= 3:
                flag = True
            else:
                anomaly_noise[rc] += 0.5

    anomaly_data = su.generate_x_from_noise(scm, anomaly_noise)
    target_score = node_score_fn(target_node, anomaly_data[target_node][0])
    assert target_score >= 3, "Anomaly not detected at the target node"
    constants.logger.info(
        f"Before weak node search, target_score: {anomaly_noise}", extra={"tags": "scm"}
    )
    return anomaly_noise


def introduce_weak_nodes(
    scm: InvertibleStructuralCausalModel,
    target_node: str,
    root_causes: list,
    linear_eqns: bool,
    invertible: bool,
):
    """
    Sample a path of weak functions from arbitrary ancestors of the target node to the target node
    """
    num_root_causes = len(root_causes)
    num_weak_nodes = np.random.randint(num_root_causes, 2 * num_root_causes)
    weak_rcs = []
    weak_nodes = []

    graph: nx.DiGraph = scm.graph
    target_ancestors = list(nx.ancestors(graph, target_node))
    target_parents = get_ordered_predecessors(graph, target_node)

    # Find non-descendants of root causes
    anchor_candidates = []
    rc_desc = set()
    for rc in root_causes:
        rc_desc.update(nx.descendants(graph, rc))
    rc_desc = rc_desc.union(set(root_causes))
    anchor_candidates = list(set(target_ancestors) - rc_desc)
    if len(anchor_candidates) > 0:
        weak_anchors = np.random.choice(sorted(anchor_candidates), num_weak_nodes)
    else:
        weak_anchors = [None] * num_weak_nodes
    weak_parents = []
    path_lengths = np.random.randint(2, 5, size=num_weak_nodes)

    def add_weak_edge(scm, linear_eqns, graph, parent, weak_node):
        su.assign_mechanism(
            node=weak_node,
            parents=parent,
            linear_eqns=linear_eqns,
            scm=scm,
            weak=not constants.WEAK_NODE_NORMAL_MECHANISM,
            invertible=invertible,
        )
        for p in parent:
            graph.add_edge(p, weak_node)
        graph.nodes[weak_node][constants.PARENTS_DURING_FIT] = get_ordered_predecessors(
            graph, weak_node
        )

    for anchor, path_length in zip(weak_anchors, path_lengths):
        parent = [anchor]
        for i in range(path_length):
            weak_node = f"S{len(weak_nodes)+1}"
            weak_nodes.append(weak_node)
            graph.add_node(weak_node)

            if parent[0] is not None:
                add_weak_edge(scm, linear_eqns, graph, parent, weak_node)
            else:
                # Add as a root node
                scm.set_causal_mechanism(
                    weak_node,
                    su.get_exogenous_dist(
                        weak=not constants.WEAK_NODE_NORMAL_MECHANISM
                    ),
                )
                graph.nodes[weak_node][
                    constants.PARENTS_DURING_FIT
                ] = get_ordered_predecessors(graph, weak_node)

            parent = [weak_node]
            if i == 1:
                weak_rcs.append(weak_node)
        weak_parents.append(parent[0])

    # Add edges from weak_parents to the target node
    for w_p in weak_parents:
        if not graph.has_edge(w_p, target_node):
            graph.add_edge(w_p, target_node)
    target_parents = get_ordered_predecessors(graph, target_node)
    graph.nodes[target_node][constants.PARENTS_DURING_FIT] = target_parents

    # Change the causal mechanism at the target
    target_causal_mechanism: AdditiveNoiseModel = scm.causal_mechanism(target_node)
    target_model = target_causal_mechanism.prediction_model.clone()
    target_model.num_weak_parents = len(weak_parents)
    target_model.num_parents = len(target_parents)
    target_model.init_weak_parameters()

    scm.set_causal_mechanism(
        target_node, AdditiveNoiseModel(target_model, su.get_exogenous_dist(weak=True))
    )

    return weak_rcs


def search_weak_rc_eps(scm, anomaly_noise, weak_rcs):
    """Searches the weak rc to be anomalouis locally"""
    normal_data = draw_samples(scm, num_samples=1000)
    means = normal_data.mean()
    stds = normal_data.std()
    score_fns = {
        node: lambda x: abs(x - means[node]) / stds[node] for node in scm.graph.nodes
    }

    for rc in weak_rcs:
        flag = False
        while flag == False:
            anomaly_data = su.generate_x_from_noise(scm, anomaly_noise)
            rc_score = score_fns[rc](anomaly_data[rc][0])
            if rc_score >= 5:
                flag = True
                constants.logger.info(f"Weak rc {rc} rc_eps: {anomaly_noise[rc]}")
            else:
                anomaly_noise[rc] += 0.5


def create_syn_testcase(
    num_root_causes,
    num_normal_data: int,
    one_cause_per_path=False,
    linear_eqns=True,
    invertible=True,
):
    """Creates one test case for Synthetic Linear Experiment
    Args:
        num_root_causes (_type_): _description_
        ground_truth_dag (_type_): _description_
        target_node (_type_): _description_
        num_samples (int, optional): _description_. Defaults to 1.
        one_cause_per_path (bool, optional): _description_. Defaults to False.
    """
    # Create the Ground Truth SCM first
    scm, target_node = su.create_scm(linear_eqns=linear_eqns, invertible=invertible)
    constants.logger.info(
        f"Total nodes in the SCM: {len(scm.graph.nodes)}", extra={"tags": "scm"}
    )

    # Draw the root causes
    root_causes = draw_root_causes(
        scm=scm,
        num_causes=num_root_causes,
        target_node=target_node,
        one_cause_per_path=one_cause_per_path,
    )

    # Sample anomalies from Q_\epsilon^{RC(j)}
    anomaly_noise = __get_rc_noise(
        scm,
        target_node,
        linear_eqns,
        root_causes,
    )

    # Introduce weak nodes
    weak_rcs = introduce_weak_nodes(
        scm=scm,
        target_node=target_node,
        root_causes=root_causes,
        linear_eqns=linear_eqns,
        invertible=invertible,
    )
    constants.logger.info(f"weak root causes: {weak_rcs}", extra={"tags": "scm"})
    constants.logger.info(
        f"Total Nodes in the SCM: {len(scm.graph.nodes)}", extra={"tags": "scm"}
    )

    for node in scm.graph.nodes:
        if node not in anomaly_noise:
            anomaly_noise[node] = constants.exog_mean_fn()

    # Make the weak root causes locally anomalous
    search_weak_rc_eps(scm, anomaly_noise, weak_rcs)

    # Generate training data and draw the anomalous sample
    normal_metrics = draw_samples(scm, num_samples=num_normal_data)
    anomaly_data = su.generate_x_from_noise(scm, anomaly_noise)

    target_score_fn = (
        lambda x: abs(x - normal_metrics.mean()[target_node])
        / normal_metrics.std()[target_node]
    )
    target_score = target_score_fn(anomaly_data[target_node][0])
    assert target_score >= 3, "Anomaly not detected at the target node"

    # First Fix all the non-weak root causes
    fix_noise = anomaly_noise.copy()
    for rc in root_causes:
        fix_noise[rc] = constants.exog_mean_fn()

    for alpha_s in cu.powerset(weak_rcs):
        weak_fix = fix_noise.copy()
        # Fix candidate weak nodes
        for a in alpha_s:
            weak_fix[a] = constants.exog_mean_fn()
        # Generate the CF
        cf_data = su.generate_x_from_noise(scm, weak_fix)
        if target_score_fn(cf_data[target_node][0]) <= 3:
            root_causes = list(root_causes) + list(alpha_s)
            fix_score = target_score_fn(cf_data[target_node][0])
            constants.logger.info(
                f"With root causes: {root_causes}, target_score: {target_score} reduced to fix score: {fix_score}",
                extra={"tags": "scm"},
            )
            break

    return {
        "scm": scm,
        "normal_metrics": normal_metrics,
        "abnormal_metrics": pd.DataFrame(anomaly_data),
        "target_node": target_node,
        "root_cause_nodes": root_causes,
    }
