# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from utils import main_utils
import rca_task as rca_task
import rca_task_syn as rca_syn
import utils.results_processing as results_processing
from utils import common_utils as cu
import constants
import time
from pathlib import Path

Path("results").mkdir(parents=True, exist_ok=True)

cu.set_seed()


def run(dataset_name, dataset_path, method_name, **kwargs):
    method_function = main_utils.get_method_function(
        method_name, linear_eqns=kwargs["linear_eqns"]
    )
    if dataset_name == "petshop":
        df = rca_task.evaluate(method_name, method_function, dataset_path)
        results_processing.save_petshop_results(method_name=method_name, df=df)
    elif dataset_name == "syn":
        df = rca_syn.evaluate_syn(
            method_name=method_name,
            method_function=method_function,
            num_root_causes=kwargs["num_root_causes"],
            one_cause_per_path=kwargs["one_cause_per_path"],
            linear_eqns=kwargs["linear_eqns"],
            invertible=kwargs["invertible"],
        )
        results_processing.save_syn_results(method_name=method_name, df=df)


def main():
    args = main_utils.parse_args()
    if args.gpu > 0:
        constants.DEVICE = f"cuda:{args.gpu}"
    dataset_name = args.dataset_name
    constants.update_anomaly_config(dataset_name, **{"linear_eqns": args.linear_eqns})
    assert args.dataset_name in ["petshop", "syn"]

    start = time.time()
    run(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        method_name=args.method,
        num_root_causes=args.num_root_causes,
        one_cause_per_path=args.one_cause_per_path,
        linear_eqns=args.linear_eqns,
        invertible=args.invertible,
    )
    end = time.time()
    if constants.logger is not None:
        constants.logger.info(f"Time taken: {(end - start) / 60} minutes")


if __name__ == "__main__":
    main()
