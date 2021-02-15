"""
Interpretability of the Deep Forest using SHAP:
    https://github.com/slundberg/shap

A helpful tutorial:
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Example%20of%20loading%20a%20custom%20tree%20model%20into%20SHAP.html#

"""

import numpy as np
import scipy
import shap
import sklearn
import graphviz

def get_shap_explainer_forest(forest):
    """
    Get the tree explainer for a forest estimator in the deep forest.

    Parameters
    ----------
    forest : :obj:`forest`
        The forest estimator that we want to explain.

    Returns
    -------
    TreeExplainer : :obj:`shap.TreeExplainer`
        Tree explainer for the forest estimator.
    """

    # Pull the info of the first tree
    children_left1     = forest.childrens[0][:, 0]
    children_right1    = forest.childrens[0][:, 1]
    children_default1  = children_right1.copy() # because sklearn does not use missing values
    features1          = forest.features[0]
    thresholds1        = forest.thresholds[0]
    values1            = forest.values[0]
    # node_sample_weight1 = forest.weighted_n_node_samples

    # Create a list of SHAP Trees
    # First we need to define a custom tree model
    tree_dicts = [
        {
            "children_left": children_left1,
            "children_right": children_right1,
            "children_default": children_default1,
            "features": features1,
            "thresholds": thresholds1,
            "values": values1 * forest.learning_rate,
            # "node_sample_weight": node_sample_weight1
        },
    ]
    model = {
        "trees": tree_dicts,
        "base_offset": scipy.special.logit(forest.init_.class_prior_[1]),
        "tree_output": "log_odds",
        "objective": "binary_crossentropy",
        "input_dtype": np.float32, # this is what type the model uses the input feature data
        "internal_dtype": np.float64 # this is what type the model uses for values and thresholds
    }

    explainer = shap.TreeExplainer(model)
