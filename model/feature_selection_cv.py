"""'The nested cross-validation and feature selection pipeline are adapted from
 https://github.com/IADI-Nancy/Sklearn_NestedCV"""

import string
import warnings
import tempfile
import numpy as np
import pandas as pd
from collections.abc import Mapping
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.base import BaseEstimator, clone
from joblib import Memory


class CV(BaseEstimator):
    """
    Abstract base class to handle Cross Validation
    """

    def __init__(
        self,
        pipeline_dic,
        params_dic,
        cv,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        pipeline_options=None,
        verbose=1,
    ):
        self.pipeline_options = pipeline_options
        self.pipeline_dic = pipeline_dic
        self.params_dic = params_dic
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.verbose = verbose

    @staticmethod
    def _string_processing(key):
        table = str.maketrans({key: "" for key in string.punctuation})
        key = key.translate(table)
        table = str.maketrans({key: "" for key in string.whitespace})
        key = key.translate(table)
        return key.lower()

    def _check_pipeline_dic(self, pipeline_dic):
        if not isinstance(pipeline_dic, Mapping):
            raise TypeError("pipeline_dic argument must be a dictionary")
        for step in pipeline_dic.keys():
            if "featureselection" in self._string_processing(step):
                if not callable(pipeline_dic[step]) and not isinstance(
                    pipeline_dic[step], str
                ):
                    raise TypeError(
                        "Dictionary values must be a callable or a string when the associated key is "
                        "DimensionalityReduction, RuleFit or FeatureSelection"
                    )
            else:
                if not callable(pipeline_dic[step]):
                    raise TypeError(
                        "Dictionary value must be a callable if associated key is not "
                        "DimensionalityReduction, RuleFit or FeatureSelection"
                    )

    def _get_parameters_grid(self, parameters_grid):
        if isinstance(parameters_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            parameters_grid = [parameters_grid]
        new_parameters_grid = []
        for grid in parameters_grid:
            parameters_dic = {}
            for step in grid.keys():
                for params in grid[step].keys():
                    parameters_dic[step + "__" + params] = grid[step][params]
            new_parameters_grid.append(parameters_dic)
        return new_parameters_grid

    def _get_pipeline(self, pipeline_dic):
        pipeline_steps = []
        for step in pipeline_dic.keys():
            kwargs = self.pipeline_options_.get(step, {})
            if not kwargs:
                warnings.warn(
                    "Default parameters are loaded for {0} (see corresponding class for detailed kwargs)".format(
                        step
                    )
                )

            if "featureselection" in self._string_processing(step):
                if self._string_processing(step) == "filterfeatureselection":
                    from model.feature_selection import FilterFeatureSelection

                    step_object = FilterFeatureSelection(pipeline_dic[step], **kwargs)
                elif self._string_processing(step) == "modelfeatureselection":
                    from model.feature_selection import ModelFeatureSelection

                    step_object = ModelFeatureSelection(pipeline_dic[step], **kwargs)
                else:
                    raise NotImplementedError(
                        self._string_processing(step), " not implemented"
                    )

            else:
                step_object = pipeline_dic[step](**kwargs)
            pipeline_steps.append((step, step_object))

        return skPipeline(pipeline_steps)

    @staticmethod
    def _check_X_Y(X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        X = np.array(X)
        assert len(X.shape) == 2, "X array must 2D"
        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.to_numpy()
            y = np.array(y)
        return X, y

    def fit(self, X, y=None, **fit_params):

        X, y = self._check_X_Y(X, y)

        if self.pipeline_options is None:
            self.pipeline_options_ = {}
        else:
            self.pipeline_options_ = dict(self.pipeline_options)
        self._check_pipeline_dic(self.pipeline_dic)
        self.pipeline = self._get_pipeline(self.pipeline_dic)
        self.params_grid = self._get_parameters_grid(self.params_dic)

        self.results = {"train": [], "test": [], "model": []}

        for k, (train_index, test_index) in enumerate(self.cv.split(X, y)):
            if self.verbose > 1:
                print(
                    "\n-----------------\n{0}/{1} <-- Current outer fold".format(
                        k + 1, self.cv.get_n_splits()
                    )
                )
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            with tempfile.TemporaryDirectory() as location:
                memory = Memory(location=location, verbose=0)
                pipeline_k = clone(self.pipeline)
                pipeline_k.set_params(memory=memory)

                pipeline_k.fit(X_train, y_train, **fit_params)

                self.append_scores(pipeline_k, train_index, test_index)
                memory.clear(warn=False)

        return self

    def append_scores(self, pipeline, train_index, test_index):
        self.results["train"].append(train_index)
        self.results["test"].append(test_index)
        self.results["model"].append(pipeline)
