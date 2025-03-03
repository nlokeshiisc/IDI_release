import random
import numpy as np
import networkx as nx
from dowhy.gcm import PredictionModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
import pandas as pd
import torch

from itertools import chain, combinations


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def simple_paths_to_target(
    graph: nx.DiGraph, target_node: str, target_ancestors: list
) -> list:
    """
    Returns a list of all simple paths from each node in target_ancestors to target_node.
    """
    # Draw atmost one cause per path
    paths = []
    for node in target_ancestors:
        node_paths = list(nx.all_simple_paths(graph, node, target_node))
        paths.extend(node_paths)

    # Remove the paths that exist in other paths as a sub-path
    paths_str = ["->".join(map(str, path)) for path in paths]
    # Get the indices of strings that are not substrings of other strings
    indices = []
    for i, ps in enumerate(paths_str):
        if not any([ps in other_ps for j, other_ps in enumerate(paths_str) if i != j]):
            indices.append(i)
    paths = [paths[i] for i in indices]
    return paths


def dict_to_df(d: dict) -> pd.DataFrame:
    try:
        return pd.DataFrame(d)
    except:
        return pd.DataFrame({k: [v] for k, v in d.items()})


def powerset(iterable):
    """Returns the powerset of the iterable

    Returns:
        _type_: _description_
    """
    s = list(iterable)
    Q = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    # sort Q in ascending order of length of the elements
    return sorted(Q, key=lambda x: len(x), reverse=False)
