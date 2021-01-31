"""A wrapper on the base estimator for the naming consistency."""


__all__ = ["Estimator"]

from .forest import RandomForestClassifier, ExtraTreesClassifier


def make_estimator(
    name,
    n_trees=100,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None
):
    # RandomForestClassifier
    if name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesClassifier
    elif name == "erf":
        estimator = ExtraTreesClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):

    def __init__(
        self,
        name,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=None,
        random_state=None
    ):
        self.estimator_ = make_estimator(name,
                                         n_trees,
                                         max_depth,
                                         min_samples_leaf,
                                         n_jobs,
                                         random_state)

    @property
    def oob_decision_function_(self):
        return self.estimator_.oob_decision_function_

    def fit_transform(self, X, y):
        self.estimator_.fit(X, y)
        X_aug = self.estimator_.oob_decision_function_

        return X_aug

    def transform(self, X):

        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict_proba(X)
