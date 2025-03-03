import matplotlib.pyplot as plt
import networkx as nx
import constants
import pydot


def plot_induced_subG(
    causal_graph,
    target_node,
    file_name,
    noise_cols=None,
    node_labels: dict = None,
):
    """
    plots the induced subgraph over the bodes
    """
    plt.clf()
    plt.cla()
    target_ancestors = nx.ancestors(causal_graph, target_node)
    subG = nx.induced_subgraph(causal_graph, target_ancestors + [target_node])

    if node_labels is not None:
        subG = nx.relabel_nodes(
            subG,
            mapping={
                node: f"{node}-{round(node_labels[node], 2)}" for node in subG.nodes
            },
            copy=True,
        )

    color_map = [
        (
            "limegreen"
            if node.split("-")[0] == target_node
            else "gold"
            if node.split("-")[0] in noise_cols
            else "skyblue"
        )
        for node in subG
    ]
    nx.draw(
        subG,
        node_color=color_map,
        with_labels=True,
    )
    plt.savefig(file_name)


def plot_nx_graph(
    graph,
    file_name,
    title="nx graph",
    target_node: str = None,
    root_causes: list = None,
    predicted_causes: list = None,
):
    """Plots the networkx graph

    Args:
        graph (_type_): _description_
        file_name (_type_): _description_
    """
    plt.clf()
    plt.cla()
    color_map = []
    for node in graph:
        if node == target_node:
            color_map.append(constants.COLOR_TGT)
        elif node in root_causes:
            color_map.append(constants.COLOR_RC)
        elif node in predicted_causes:
            color_map.append(constants.COLOR_RCPRED)
        elif node[0] == "S":
            color_map.append(constants.COLOR_WEAK)  # This is a weak node
        else:
            color_map.append(constants.COLOR_NRM)

    plt.clf()
    plt.cla()
    plt.figure(figsize=(20, 14))
    try:
        nx.draw(
            graph,
            pos=nx.nx_pydot.pydot_layout(
                graph, prog="dot"
            ),  # nx.nx_agraph.graphviz_layout(graph, prog="dot"),
            node_size=3000,
            linewidths=1.5,
            font_size=25,
            font_weight="bold",
            with_labels=True,
            node_color=color_map,
        )
    except:
        nx.draw(
            graph,
            node_size=3000,
            linewidths=1.5,
            font_size=25,
            font_weight="bold",
            with_labels=True,
            node_color=color_map,
        )
    plt.title(title)
    plt.savefig(constants.RESULTS_DIR / "graph" / file_name)
