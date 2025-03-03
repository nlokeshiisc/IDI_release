# In-Distribution Interventions for Root Cause Analysis.

| [Accepted at ICLR 25](https://openreview.net/forum?id=l11DZY5Nxu&referrer=%5Bthe%20profile%20of%20Lokesh%20Nagalapatti%5D(%2Fprofile%3Fid%3D~Lokesh_Nagalapatti1)).  

# Code

All our code is implemented within the [Petshop library](https://github.com/amazon-science/petshop-root-cause-analysis) for Root Cause Diagnosis.

Please install the packages in the file: [requirements.txt](requirements.txt)

---

## IDI Code

IDI code is available in [idint.py](methods/idint.py)
The code for IDI method in CF mode is available in [oodcf.py](methods/oodcf.py)

The only code change between these two methods is:
```python
if cf_rca == False:
    int_samples = interventional_samples(
        causal_model=learned_scm,
        interventions=int_dict,
        observed_data=abnormal_metrics,
    )
else:
    int_samples = counterfactual_samples(
        causal_model=learned_scm,
        interventions=int_dict,
        observed_data=abnormal_metrics,
    )
```

Whether we use int, or CF for assessing the causal effect of fix at $X_n$ 



---

We provide additional implementation for `smooth traversal`, and `toca`.

---

# Toy Experiments

We provide the implementation for the toy experiments in the following Jupiter notebooks.

1. [Linear and Non-Linear four variable toy SCMs](limitation.ipynb)
2. [Experiments showing impact of correlated covariates on the Linear toy SCM](limitation_linear.ipynb)
3. [Experiments showing impact of the depth between root cause node and the anomalous nodes](limitation_linear.ipynb)


# Running Code

We added a [python script](script.py) that reproduces all the results in our paper. the command to run is as follows:

```
python script.py --dataset_name <ds> --num_rcs <numrcs> --ocpp <ocpp> --linear_eqns <lin> --method <method> --invertible <inv>
```

where:

1. dataset_name -- should be `petshop` or `syn`
> The following arguments are relevant only for Synthetic Experiments
2. numrcs -- should be 1 for unique root cause, else set as required. We used `3` for Multiple RCs
3. ocpp -- set to true means there is atmost one RC in a simple path that leads to the target $X_n$
4. lin -- set to true gives a Linear SCM, else an MLP
5. method -- set to `all` to run all methods
6. invertible -- set to true, gives an Additive Noise MLP Oracle SCM, false gives an MLP that takes in $\epsilon$ also as input.