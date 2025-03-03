import argparse


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="A script to perform a anomaly-traversal to identify root-causes based on performance metrics from the PetShop application.")
    parser.add_argument("--dataset_name", default="syn", type=str, help="Dataset Name; options petshop, syn")
    parser.add_argument("--dataset_path",default="dataset", type=str, help="Path to petshop metric dataset.")
    parser.add_argument("--method", default="idint", type=str, help="baseline, rcd, epsilon_diagnosis, counterfactual_attribution, circa and ranked_correlation, idint currently implemented.")
    parser.add_argument("--num_root_causes",  default=3, type=int, help="Number of root causes to inject into the synthetic test cases.")
    
    parser.add_argument("--one_cause_per_path", action="store_true", dest="one_cause_per_path")
    parser.add_argument("--no-one_cause_per_path", dest="one_cause_per_path",  action="store_false")
    parser.set_defaults(one_cause_per_path=True)
    parser.add_argument("--linear_eqns", action="store_true", dest="linear_eqns")
    parser.add_argument("--no-linear_eqns",  dest="linear_eqns",  action="store_false")
    parser.set_defaults(linear_eqns=False)
    parser.add_argument("--invertible", action="store_true", dest="invertible")
    parser.add_argument("--no-invertible",  dest="invertible",  action="store_false")
    parser.set_defaults(invertible=True)
    
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID to use.")
    return parser.parse_args()
    # fmt: on


def get_method_function(method_name, linear_eqns: bool = False):
    if method_name == "traversal":
        from methods import baseline_anomaly_traversal

        method = baseline_anomaly_traversal.make_traversal()

    elif method_name == "rcd":
        from methods import hierarchical_rcd

        method = hierarchical_rcd.make_hierarchical_rcd()

    elif method_name == "epsilon_diagnosis":
        from methods import epsilon_diagnosis

        method = epsilon_diagnosis.make_epsilon_diagnosis()

    elif method_name == "circa":
        from methods import circa

        method = circa.make_circa()

    elif method_name == "ranked_correlation":
        from methods import ranked_correlation

        method = ranked_correlation.make_ranked_correlation()

    elif method_name == "random_walk":
        from methods import random_walk

        method = random_walk.make_random_walk()

    elif method_name == "counterfactual_attribution":
        # This is the Shapley method
        from methods import counterfactual_attribution

        method = counterfactual_attribution.make_counterfactual_attribution_method(
            linear_eqns=linear_eqns
        )

    elif method_name == "hierarchical_rcd":
        from methods import hierarchical_rcd

        method = hierarchical_rcd.make_hierarchical_rcd()

    elif method_name == "idint":
        from methods import idint

        method = idint.make_idint(linear_eqns=linear_eqns)

    elif method_name == "oodcf":
        from methods import oodcf

        method = oodcf.make_ood_cf(linear_eqns=linear_eqns)

    # TODO: Implement the following methods
    elif method_name == "toca":
        from methods import toca

        method = toca.make_toca(linear_eqns=linear_eqns)

    elif method_name == "smooth_traversal":
        from methods import smooth_traversal

        method = smooth_traversal.make_smooth_traversal()

    else:
        raise ValueError(f"Unsupported method {method_name}")

    return method
