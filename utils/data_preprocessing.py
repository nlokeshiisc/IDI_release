# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pandas as pd
from typing import Set
import numpy as np


# === BASIC PREPROCESSING === #

def map_df(df):
    """
    
    """
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new


def reduce_df(df: pd.DataFrame, metric: str, statistic: str):
    data_matrix = df.loc[:, (slice(None), [metric], [statistic])]
    # Now we can focus on the microservices as only column since metric and statistic are fixed.
    data_matrix.columns = [c[0] for c in data_matrix.columns]
    return data_matrix


def marginalize_node(graph: nx.DiGraph, node: str):
    children = graph.successors(node)
    for child in children:
        graph.add_edges_from((n, child) for n in graph.predecessors(node))
    graph.remove_node(node)


def impute_df(df: pd.DataFrame, method: str = "mean", fill: float = -1):
    """
    Wrapper around very simple imputation methods.

    Args:
        df: Pandas DataFrame in which to impute NaNs.
        method: How NaNs should be imputed. If 'mean' then each is replaced by the mean of the
            remaining values of the same microservice, metric and statistic. If 'interpolate' then
            pandas.DataFrame.interpolate(method='time',limit_direction='both') is used.
            if 'fill' then missing values will be replaced with the value `fill`.
        fill: Value with which to replace NaNs if `method = 'fill'`.
    """
    if method not in {"mean", "interpolate", "fill"}:
        ValueError(f"{method} is not a valid imputation method.")
    if method == "mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "interpolate":
        df_index = df.index
        df.index = pd.to_datetime(df.index, unit="s")
        df.interpolate("time", limit_direction="both", inplace=True)
        df.interpolate("time", limit_direction="both", inplace=True)
        # reverting index back for consistency between imputation methods
        df.index = df_index
    elif method == "fill":
        df.fillna(fill, inplace=True)
        
def pad_and_fill(data_matrix: pd.DataFrame, fill_df: pd.DataFrame):
    original_columns = data_matrix.columns
    overall_mean = np.nanmean(data_matrix.mean())
    for c in fill_df.columns:
        if c not in data_matrix.columns:
            data_matrix[c] = fill_df[c].mean()
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    data_matrix.fillna(overall_mean, inplace=True)
    return data_matrix, original_columns


def pad_and_replace_nan(data_matrix: pd.DataFrame, required_columns: Set[str]):
    # TODO: Cleanup
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    overall_mean = np.nanmean(data_matrix.mean())
    data_matrix.fillna(overall_mean, inplace=True)
    for c in set(required_columns) - set(data_matrix.columns):
        data_matrix[c] = overall_mean
    return data_matrix