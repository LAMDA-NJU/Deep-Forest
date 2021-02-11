"""A wrapper on the base estimator for the naming consistency."""


__all__ = ["Estimator"]

from .forest import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)


def make_classifier_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
):
    # RandomForestClassifier
    if name == "rf":
        estimator = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesClassifier
    elif name == "erf":
        estimator = ExtraTreesClassifier(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


def make_regressor_estimator(
    name,
    criterion,
    n_trees=100,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=None,
    random_state=None,
):
    # RandomForestRegressor
    if name == "rf":
        estimator = RandomForestRegressor(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    # ExtraTreesRegressor
    elif name == "erf":
        estimator = ExtraTreesRegressor(
            criterion=criterion,
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        msg = "Unknown type of estimator, which should be one of {{rf, erf}}."
        raise NotImplementedError(msg)

    return estimator


class Estimator(object):
    def __init__(
        self,
        name,
        criterion,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=None,
        random_state=None,
        is_classifier=True,
    ):

        self.is_classifier = is_classifier
        if self.is_classifier:
            self.estimator_ = make_classifier_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_leaf,
                n_jobs,
                random_state,
            )
        else:
            self.estimator_ = make_regressor_estimator(
                name,
                criterion,
                n_trees,
                max_depth,
                min_samples_leaf,
                n_jobs,
                random_state,
            )

    @property
    def oob_decision_function_(self):
        return self.estimator_.oob_decision_function_

    def fit_transform(self, X, y, sample_weight=None):
        self.estimator_.fit(X, y, sample_weight)
        X_aug = self.estimator_.oob_decision_function_

        return X_aug

    def transform(self, X):
        if self.is_classifier:
            return self.estimator_.predict_proba(X)
        return self.estimator_.predict(X)

    def predict(self, X):
        if self.is_classifier:
            return self.estimator_.predict_proba(X)
        return self.estimator_.predict(X)
