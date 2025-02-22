"""'The nested cross-validation and feature selection pipeline are adapted from
 https://github.com/IADI-Nancy/Sklearn_NestedCV"""

import string
import warnings
import tempfile
import numpy as np
import pandas as pd
from collections.abc import Mapping
from sklearn.pipeline import Pipeline as skPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics._scorer import check_scoring, _check_multimetric_scoring
from sklearn.metrics import (
    make_scorer,
    f1_score,
    fbeta_score,
    brier_score_loss,
    log_loss,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, clone
from joblib import Memory
from shutil import rmtree

# from .dimensionality_reduction import DimensionalityReduction
from .feature_selection import (
    FilterFeatureSelection,
    ModelFeatureSelection,
)  # , BorutaShapFeatureSelection


def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    return proxied_func(y_true, y_probs, **kwargs)


class NestedCV(BaseEstimator):
    """
    Abstract base class to handle Nested Cross Validation with rule fit support
    """

    def __init__(
        self,
        pipeline_dic,
        params_dic,
        outer_cv=5,
        inner_cv=5,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        imblearn_pipeline=False,
        pipeline_options=None,
        metric="roc_auc",
        verbose=1,
        refit_outer=True,
        error_score=np.nan,
        refit_inner=True,
        return_train_score=False,
    ):
        self.imblearn_pipeline = imblearn_pipeline
        self.pipeline_options = pipeline_options
        self.pipeline_dic = pipeline_dic
        self.params_dic = params_dic
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.metric = metric
        self.verbose = verbose
        self.refit_outer = refit_outer
        self.error_score = error_score
        self.refit_inner = refit_inner
        self.return_train_score = return_train_score

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
            if (
                "dimensionalityreduction" in self._string_processing(step)
                or "rulefit" in self._string_processing(step)
                or "featureselection" in self._string_processing(step)
            ):
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
                    if self._string_processing(step) == "dimensionalityreduction":
                        parameters_dic[step + "__method__" + params] = grid[step][
                            params
                        ]
                    else:
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
            if "dimensionalityreduction" in self._string_processing(step):
                raise NotImplementedError(
                    self._string_processing(step), " not implemented"
                )

            elif "featureselection" in self._string_processing(step):
                if self._string_processing(step) == "filterfeatureselection":
                    step_object = FilterFeatureSelection(pipeline_dic[step], **kwargs)
                elif self._string_processing(step) == "modelfeatureselection":
                    step_object = ModelFeatureSelection(pipeline_dic[step], **kwargs)
                # elif self._string_processing(step) == 'borutashapfeatureselection':
                #     step_object = BorutaShapFeatureSelection(pipeline_dic[step], **kwargs)
                else:
                    raise NotImplementedError(
                        self._string_processing(step), " not implemented"
                    )

            else:
                step_object = pipeline_dic[step](**kwargs)
            pipeline_steps.append((step, step_object))

        if self.imblearn_pipeline:
            return imbPipeline(pipeline_steps)
        else:
            return skPipeline(pipeline_steps)

    def _check_is_fitted(self, method_name):
        if not self.refit_outer:
            raise NotFittedError(
                "This %s instance was initialized "
                "with refit=False. %s is "
                "available only after refitting on the best "
                "parameters. You can refit an estimator "
                "manually using the ``best_params_`` "
                "attribute" % (type(self).__name__, method_name)
            )
        else:
            check_is_fitted(self)

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

    def _check_metric(self):

        if self.metric == "f1_score":
            self.metric = make_scorer(f1_score, average="binary", pos_label=1)

        if self.metric == "fbeta_score":
            self.metric = make_scorer(
                fbeta_score, beta=1.5, average="binary", pos_label=1
            )

        if self.metric == "brier_score":
            self.metric = make_scorer(
                ProbaScoreProxy,
                greater_is_better=False,
                needs_proba=True,
                class_idx=1,
                proxied_func=brier_score_loss,
            )

        if self.metric == "log_loss":
            self.metric = make_scorer(
                ProbaScoreProxy,
                greater_is_better=False,
                needs_proba=True,
                class_idx=1,
                proxied_func=log_loss,
            )

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        raise NotImplementedError("_get_inner_param_optimizer not implemented")

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        raise NotImplementedError("_get_outer_param_optimizer not implemented")

    def _check_refit_for_multimetric(self):
        if self.multimetric_:
            if callable(self.refit_inner):
                raise ValueError(
                    "If inner loops use multimetric scoring and the user want to refit according to a "
                    "callable, the latter must be passed in a dictionary {score: callable} with score "
                    "being the score name with which the score on different sets will be calculated"
                )
            if self.refit_inner is not False and (
                not isinstance(self.refit_inner, str)
                or
                # This will work for both dict / list (tuple)
                self.refit_inner not in self.scorers
            ):
                if isinstance(self.refit_inner, Mapping):
                    if len(self.refit_inner.keys()) > 1:
                        raise ValueError(
                            "refit_inner dict must have only one key, got %d"
                            % len(self.refit_inner.keys())
                        )
                    self.refit_metric = list(self.refit_inner.keys())[0]
                    self.refit_inner = self.refit_inner[self.refit_metric]
                else:
                    raise ValueError(
                        "For multi-metric scoring, the parameter "
                        "refit must be set to a scorer key or a "
                        "dict with scorer key and callable value to refit an estimator with the "
                        "best parameter setting on the whole "
                        "data and make the best_* attributes "
                        "available for that metric. If this is "
                        "not needed, refit should be set to "
                        "False explicitly. %r was passed." % self.refit_inner
                    )
            else:
                self.refit_metric = self.refit_inner
        else:
            self.refit_metric = "score"
            if self.refit_inner is True:
                self.refit_inner = "score"

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fit Nested CV with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        Returns
            -------
            It will not return directly the values, but it's accessable from the class object it self.
            You should be able to access:
            outer_pred
                 A dictionary to access the train indexes, the test indexes and the model  of each outer loop
                 for further post-processing. Keys are respectively train, test and model with values being
                 lists of length outer_cv.get_n_splits().
            outer_results
                A dictionary to access the outer test scores, the best inner scores, the best inner parameters (and
                outer_train_scores if return_train_score == True). Keys are respectively outer_test_score,
                best_inner_score, best_inner_params (and outer_train_score) with values being lists of length
                outer_cv.get_n_splits().
            inner_results
                A list of dictionary of length outer_cv.get_n_splits().
                Each dictionary having params, mean_test_score, std_test_score (and mean_train_score, std_train_score
                if return_train_score == True) as keys and values being the list of params or associated results
                over the inner loops.
            best_estimator_
                Model when refit on the whole dataset with hyperparameter optimized by GridSearch CV.
                Available only if refit == True.
        """
        X, y = self._check_X_Y(X, y)

        if self.pipeline_options is None:
            self.pipeline_options_ = {}
        else:
            self.pipeline_options_ = dict(self.pipeline_options)
        self._check_pipeline_dic(self.pipeline_dic)
        self.model = self._get_pipeline(self.pipeline_dic)
        self.params_grid = self._get_parameters_grid(self.params_dic)

        # Last element of pipeline = estimator
        outer_cv = check_cv(self.outer_cv, y, classifier=is_classifier(self.model[-1]))
        inner_cv = check_cv(self.inner_cv, y, classifier=is_classifier(self.model[-1]))

        self.outer_pred = {
            "train": [],
            "test": [],
            "model": [],
            "predict_train": [],
            "predict_test": [],
        }
        if hasattr(self.model[-1], "predict_proba"):
            self.outer_pred.update(
                {"predict_proba_train": [], "predict_proba_test": []}
            )
        if hasattr(self.model[-1], "decision_function"):
            self.outer_pred.update(
                {"decision_function_train": [], "decision_function_test": []}
            )

        self.outer_results = {
            "outer_test_score": [],
            "inner_test_score": [],
            "best_inner_score": [],
            "best_inner_params": [],
        }
        self.inner_results = []

        if self.return_train_score:
            self.outer_results.update(
                {"outer_train_score": [], "inner_train_score": []}
            )

        if self.refit_outer:
            self.refit_outer_pred = {
                "train": [],
                "test": [],
                "model": [],
                "predict_train": [],
                "predict_test": [],
            }
            if hasattr(self.model[-1], "predict_proba"):
                self.refit_outer_pred.update(
                    {"predict_proba_train": [], "predict_proba_test": []}
                )
            if hasattr(self.model[-1], "decision_function"):
                self.refit_outer_pred.update(
                    {"decision_function_train": [], "decision_function_test": []}
                )

            self.refit_outer_results = {
                "outer_test_score": [],
                "best_inner_score": [],
                "best_inner_params": [],
            }
            self.refit_inner_results = []

            if self.return_train_score:
                self.refit_outer_results.update({"outer_train_score": []})

        # From sklearn.model_selection._search.BasesearchCV
        if not callable(self.metric):
            self._check_metric()

        if callable(self.metric) or self.metric is None or isinstance(self.metric, str):
            self.scorers = {"score": check_scoring(self.model, scoring=self.metric)}
            self.multimetric_ = False
        else:
            self.scorers = _check_multimetric_scoring(self.model, scoring=self.metric)
            self.multimetric_ = True

        self._check_refit_for_multimetric()

        # outer cross-val loop
        for k_outer, (train_outer_index, test_outer_index) in enumerate(
            outer_cv.split(X, y, groups)
        ):
            if self.verbose > 1:
                print(
                    "\n-----------------\n{0}/{1} <-- Current outer fold".format(
                        k_outer + 1, outer_cv.get_n_splits()
                    )
                )
            X_train_outer, X_test_outer = X[train_outer_index], X[test_outer_index]
            y_train_outer, y_test_outer = y[train_outer_index], y[test_outer_index]
            with tempfile.TemporaryDirectory() as location:
                memory = Memory(location=location, verbose=0)
                inner_model = clone(self.model)
                inner_model.set_params(memory=memory)

                # inner cross-val loop: fit classifier for different parameter configurations
                pipeline_inner = self._get_inner_param_optimizer(inner_model, inner_cv)
                pipeline_inner.fit(
                    X_train_outer, y_train_outer, groups=groups, **fit_params
                )

                # store the intermediate inner cross-val results
                self.append_scores(
                    pipeline_inner,
                    X_train_outer,
                    X_test_outer,
                    y_train_outer,
                    y_test_outer,
                    train_outer_index,
                    test_outer_index,
                )

                if self.verbose > 1:
                    print(
                        "\nResults for outer fold:\nBest inner parameters was: {0}".format(
                            self.outer_results["best_inner_params"][-1]
                        )
                    )
                    print(
                        "Outer test score: {0}".format(
                            self.outer_results["outer_test_score"][-1]
                        )
                    )

                    mean_test_score = pipeline_inner.cv_results_[
                        "mean_test_%s" % self.refit_metric
                    ]
                    mean_train_score = pipeline_inner.cv_results_[
                        "mean_train_%s" % self.refit_metric
                    ]

                    for params_dict in pipeline_inner.cv_results_["params"]:
                        index_params_dic = pipeline_inner.cv_results_["params"].index(
                            params_dict
                        )
                        if self.return_train_score:
                            print(
                                "\t\t Params: {0}, Mean inner test score: {1}, Mean inner train score: {2}".format(
                                    params_dict,
                                    mean_test_score[index_params_dic],
                                    mean_train_score[index_params_dic],
                                )
                            )
                        else:
                            print(
                                "\t\t Params: {0}, Mean inner test score: {1}".format(
                                    params_dict, mean_test_score[index_params_dic]
                                )
                            )

                    print(
                        "Best inner score: {0}".format(
                            self.outer_results["best_inner_score"][-1]
                        )
                    )

                    if self.return_train_score:
                        print(
                            "Outer train score: {0}".format(
                                self.outer_results["outer_train_score"][-1]
                            )
                        )

                memory.clear(warn=False)

        if self.verbose > 0:
            print(
                "\nOverall outer test score (mean +/- std): {0} +/- {1}".format(
                    np.mean(self.outer_results["outer_test_score"]),
                    np.std(self.outer_results["outer_test_score"]),
                )
            )
            print("Best params by outer fold:")
            for i, params_dict in enumerate(self.outer_results["best_inner_params"]):
                print("\t Outer fold {0}: {1}".format(i + 1, params_dict))
            print("\n")

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = self.scorers if self.multimetric_ else self.scorers["score"]

        # If refit is True Hyperparameter optimization on whole dataset and fit with best params
        if self.refit_outer:
            print("=== Refit ===")
            location = "cachedir"
            memory = Memory(location=location, verbose=0)
            final_model = clone(self.model)
            final_model.set_params(memory=memory)
            pipeline_refit = self._get_outer_param_optimizer(final_model, outer_cv)
            pipeline_refit.fit(X, y, groups=groups, **fit_params)
            self.refit_scores(pipeline_refit, X, y)
            self.best_estimator_ = pipeline_refit.best_estimator_

            memory.clear(warn=False)
            rmtree(location)

            if self.verbose > 0:
                print(
                    "\nRefitted model train score (mean +/- std): {0} +/- {1}".format(
                        np.mean(self.refit_outer_results["outer_train_score"]),
                        np.std(self.refit_outer_results["outer_train_score"]),
                    )
                )
                print("Best params:")
                for i, params_dict in enumerate(
                    self.refit_outer_results["best_inner_params"]
                ):
                    print("\t Final model {0}: {1}".format(i + 1, params_dict))
                print("\n")

        return self

    # Append inner cross-validation results summary for all outer folds (--> called n_repeats_outer*n_folds_outer times)
    # The result table summarizes outer_results, inner_results
    def append_scores(
        self,
        pipeline_inner,
        X_train_outer,
        X_test_outer,
        y_train_outer,
        y_test_outer,
        train_outer_index,
        test_outer_index,
    ):

        self.inner_results.append(
            {
                "params": pipeline_inner.cv_results_["params"],
                "mean_test_score": pipeline_inner.cv_results_[
                    "mean_test_%s" % self.refit_metric
                ],
                "std_test_score": pipeline_inner.cv_results_[
                    "std_test_%s" % self.refit_metric
                ],
            }
        )
        if self.return_train_score:
            self.inner_results[-1].update(
                {
                    "mean_train_score": pipeline_inner.cv_results_[
                        "mean_train_%s" % self.refit_metric
                    ],
                    "std_train_score": pipeline_inner.cv_results_[
                        "std_train_%s" % self.refit_metric
                    ],
                }
            )
        self.outer_results["best_inner_score"].append(
            pipeline_inner.cv_results_["mean_test_%s" % self.refit_metric][
                pipeline_inner.best_index_
            ]
        )  # Because best_score doesn't exist if refit_inner is a callable
        self.outer_results["best_inner_params"].append(pipeline_inner.best_params_)
        if self.return_train_score:
            self.outer_results["outer_train_score"].append(
                self.scorers[self.refit_metric](
                    pipeline_inner.best_estimator_, X_train_outer, y_train_outer
                )
            )

        self.outer_results["outer_test_score"].append(
            self.scorers[self.refit_metric](
                pipeline_inner.best_estimator_, X_test_outer, y_test_outer
            )
        )
        self.outer_pred["train"].append(train_outer_index)
        self.outer_pred["test"].append(test_outer_index)
        self.outer_pred["model"].append(pipeline_inner.best_estimator_)
        self.outer_pred["predict_train"].append(
            pipeline_inner.best_estimator_.predict(X_train_outer)
        )
        self.outer_pred["predict_test"].append(
            pipeline_inner.best_estimator_.predict(X_test_outer)
        )
        if hasattr(pipeline_inner.best_estimator_[-1], "predict_proba"):
            self.outer_pred["predict_proba_train"].append(
                pipeline_inner.best_estimator_.predict_proba(X_train_outer)
            )
            self.outer_pred["predict_proba_test"].append(
                pipeline_inner.best_estimator_.predict_proba(X_test_outer)
            )
        if hasattr(pipeline_inner.best_estimator_[-1], "decision_function"):
            self.outer_pred["decision_function_train"].append(
                pipeline_inner.best_estimator_.decision_function(X_train_outer)
            )
            self.outer_pred["decision_function_test"].append(
                pipeline_inner.best_estimator_.decision_function(X_test_outer)
            )

    # Refit the model on the entire dataset, can be used for external test set evaluation
    def refit_scores(self, pipeline_refit, X, y):
        self.refit_inner_results.append(
            {
                "params": pipeline_refit.cv_results_["params"],
                "mean_test_score": pipeline_refit.cv_results_[
                    "mean_test_%s" % self.refit_metric
                ],
                "std_test_score": pipeline_refit.cv_results_[
                    "std_test_%s" % self.refit_metric
                ],
            }
        )
        if self.return_train_score:
            self.refit_inner_results[-1].update(
                {
                    "mean_train_score": pipeline_refit.cv_results_[
                        "mean_train_%s" % self.refit_metric
                    ],
                    "std_train_score": pipeline_refit.cv_results_[
                        "std_train_%s" % self.refit_metric
                    ],
                }
            )
        self.refit_outer_results["best_inner_score"].append(
            pipeline_refit.cv_results_["mean_test_%s" % self.refit_metric][
                pipeline_refit.best_index_
            ]
        )  # Because best_score doesn't exist if refit_inner is a callable
        self.refit_outer_results["best_inner_params"].append(
            pipeline_refit.best_params_
        )
        if self.return_train_score:
            self.refit_outer_results["outer_train_score"].append(
                self.scorers[self.refit_metric](pipeline_refit.best_estimator_, X, y)
            )

        self.refit_outer_pred["model"].append(pipeline_refit.best_estimator_)
        self.refit_outer_pred["predict_train"].append(
            pipeline_refit.best_estimator_.predict(X)
        )
        if hasattr(pipeline_refit.best_estimator_[-1], "predict_proba"):
            self.refit_outer_pred["predict_proba_train"].append(
                pipeline_refit.best_estimator_.predict_proba(X)
            )
        if hasattr(pipeline_refit.best_estimator_[-1], "decision_function"):
            self.refit_outer_pred["decision_function_train"].append(
                pipeline_refit.best_estimator_.decision_function(X)
            )

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples, n_output) or (n_samples,), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted("score")
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )
        score = self.scorer_[self.refit_metric] if self.multimetric_ else self.scorer_
        return score(self.best_estimator_, X, y)

    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict_proba")
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("predict_log_proba")
        return self.best_estimator_.predict_log_proba(X)

    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("decision_function")
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("transform")
        return self.best_estimator_.transform(X)

    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted("inverse_transform")
        return self.best_estimator_.inverse_transform(Xt)

    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_


class GridSearchNestedCV(NestedCV):
    """
    Nested Cross Validation  with grid search hyperparameter optimization in inner loop
    Parameters
    ----------
    pipeline_dic: dictionary {str: callable}
        Dictionary containing the steps with which the pipeline will be constructed. Pipeline steps names (string) as
        keys and sklearn-like transform object (callable) as value except the last object that must be
        an sklearn-like estimator (callable).
        Steps will be chained in the order in which they are given.
        If some steps include callable from imblearn package
        (https://imbalanced-learn.readthedocs.io/en/stable/index.html) imblearn_pipeline option must be set to True (see
        cv_options argument).
        If key is either 'FeatureSelection' or 'DimensionalityReduction' the value can be either str or callable.
        Keyword options of each callable can be given in cv_options (see cv_options argument)
        Example:
        pipeline_dic = {'scale': sklearn.preprocessing.StandardScaler,
                        'oversampling': imblearn.over_sampling.SMOTE,
                        'DimensionalityReduction': sklearn.decomposition.PCA,
                        'FeatureSelection': 'mw',
                        'classifier': sklearn.linear_model.LogisiticRegression}
    params_grid: dict or list of dictionaries
        Dictionary with step names (string) as keys and dictionary with parameters grid as values,
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.
        Parameters grid given as value must be dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values as taken by sklearn.model_selection.GridSearchCV.
        Example with previous pipeline:
        params_grid = [{'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['l1'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga']}},
                       {'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['elasticnet'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga'],
                                       'l1_ratio': np.arange(0.1, 1, 0.1)}}]
    outer_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    inner_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    n_jobs: int or None, optional (default=None)
        Number of jobs to run in parallel in inner loop. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
        See sklearn.model_selection.GridSearchCV for more details.
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful
        to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            -None, in which case all the jobs are immediately created and spawned. Use this for lightweight and
            fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            -An int, giving the exact number of total jobs that are spawned
            -A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
        See sklearn.model_selection.GridSearchCV for more details
    imblearn_pipeline: boolean (default=False)
        Indicate whether callable from imblearn package are used in pipeline
    pipeline_options: dict (default={})
        Dictionary with step names (string) as key and dictionary of the associated keywords as values. These
        keywords will be used to construct the object while creating the pipeline. If not specified, the object will
        be constructed with default parameters.
        Example with previous pipeline:
        pipeline_options = {'oversampling': {'sampling_strategy': 'minority_class'},
                            'DimensionalityReduction': {'n_components': 0.95},
                            'FeatureSelection': {'bootstrap': True, 'n_bsamples': 200, 'n_selected_features': 10,
                                                 'ranking_aggregation': 'importance_score'}}
    metric: string, callable or None, (default='roc_auc')
        A single string (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        or a callable (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) to evaluate the
        predictions on the test set.
        If None, the estimator’s score method is used.
        Multimetric scoring is available for inner loops. If multimetric, outer loop will only be score with metric
        specified by refit_inner
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric (default=np.nan)
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which
        will always raise the error.
    refit_inner: boolean, string or callable (default=True)
        Refit an estimator using the best found parameters on the whole outer training set
        Argument will be given to GridsearchCV that select hyperparameters in the inner loop :
            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the
            best parameters for refitting the estimator at the end. Where there are considerations other than maximum
            score in choosing a best estimator, refit can be set to a function which returns the selected best_index_
            given cv_results_. In that case, the best_estimator_ and best_parameters_ will be set according to the
            returned best_index_. The refitted estimator is made available at the best_estimator_ attribute and permits
            using predict directly on this GridSearchCV instance. If inner loops use multimetric scoring and the user
            want to refit according to a callable, the latter must be passed in a dictionary {score: callable} with
            score being the score name with which the score on different sets will be calculated
    refit_outer: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a gridsearch cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    """

    def __init__(
        self,
        pipeline_dic,
        params_dic,
        outer_cv=5,
        inner_cv=5,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        imblearn_pipeline=False,
        pipeline_options=None,
        metric="roc_auc",
        verbose=1,
        refit_outer=True,
        error_score=np.nan,
        refit_inner=True,
        return_train_score=False,
    ):
        super().__init__(
            pipeline_dic,
            params_dic,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            imblearn_pipeline=imblearn_pipeline,
            pipeline_options=pipeline_options,
            metric=metric,
            verbose=verbose,
            refit_outer=refit_outer,
            error_score=error_score,
            refit_inner=refit_inner,
            return_train_score=return_train_score,
        )

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        return GridSearchCV(
            inner_model,
            self.params_grid,
            scoring=self.scorers,
            n_jobs=self.n_jobs,
            cv=inner_cv,
            return_train_score=self.return_train_score,
            verbose=self.verbose - 1,
            pre_dispatch=self.pre_dispatch,
            refit=self.refit_inner,
            error_score=self.error_score,
        )

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        return GridSearchCV(
            final_model,
            self.params_grid,
            scoring=self.scorers[self.refit_metric],
            n_jobs=self.n_jobs,
            cv=outer_cv,
            return_train_score=self.return_train_score,
            verbose=self.verbose - 1,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )


class RandomSearchNestedCV(NestedCV):
    """
    Nested Cross Validation with random search hyperparameter optimization in inner loop
    Parameters
    ----------
    pipeline_dic: dictionary {str: callable}
        Dictionary containing the steps with which the pipeline will be constructed. Pipeline steps names (string) as
        keys and sklearn-like transform object (callable) as value except the last object that must be
        an sklearn-like estimator (callable).
        Steps will be chained in the order in which they are given.
        If some steps include callable from imblearn package
        (https://imbalanced-learn.readthedocs.io/en/stable/index.html) imblearn_pipeline option must be set to True (see
        cv_options argument).
        If key is either 'FeatureSelection' or 'DimensionalityReduction' the value can be either str or callable.
        Keyword options of each callable can be given in cv_options (see cv_options argument)
        Example:
        pipeline_dic = {'scale': sklearn.preprocessing.StandardScaler,
                        'oversampling': imblearn.over_sampling.SMOTE,
                        'DimensionalityReduction': sklearn.decomposition.PCA,
                        'FeatureSelection': 'mw',
                        'classifier': sklearn.linear_model.LogisiticRegression}
    params_grid: dict or list of dictionaries
        Dictionary with step names (string) as keys and dictionary with parameters grid as values,
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
        This enables searching over any sequence of parameter settings.
        Parameters grid given as value must be dictionary with parameters names (string) as keys and lists of parameter
        settings to try as values as taken by sklearn.model_selection.GridSearchCV.
        Example with previous pipeline:
        params_grid = [{'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['l1'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga']}},
                       {'DimensionalityReduction': {'n_components': [0.95, 0.99], 'svd_solver': ['full']},
                        'FeatureSelection': {'n_selected_features': [5,10,15,20,n_features]},
                        'classifier': {'penalty': ['elasticnet'], 'C': np.arange(0.001, 1, 0.002), 'solver': ['saga'],
                                       'l1_ratio': np.arange(0.1, 1, 0.1)}}]
    outer_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    inner_cv: int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            None, to use the default 5-fold cross validation,
            integer, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        See sklearn.model_selection.GridSearchCV for more details.
    n_jobs: int or None, optional (default=None)
        Number of jobs to run in parallel in inner loop. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
        See sklearn.model_selection.GridSearchCV for more details.
    pre_dispatch: int, or string, optional
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful
        to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            -None, in which case all the jobs are immediately created and spawned. Use this for lightweight and
            fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            -An int, giving the exact number of total jobs that are spawned
            -A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
        See sklearn.model_selection.GridSearchCV for more details
    imblearn_pipeline: boolean (default=False)
        Indicate whether callable from imblearn package are used in pipeline
    pipeline_options: dict (default={})
        Dictionary with step names (string) as key and dictionary of the associated keywords as values. These
        keywords will be used to construct the object while creating the pipeline. If not specified, the object will
        be constructed with default parameters.
        Example with previous pipeline:
        pipeline_options = {'oversampling': {'sampling_strategy': 'minority_class'},
                            'DimensionalityReduction': {'n_components': 0.95},
                            'FeatureSelection': {'bootstrap': True, 'n_bsamples': 200, 'n_selected_features': 10,
                                                 'ranking_aggregation': 'importance_score'}}
    metric: string, callable or None, (default='roc_auc')
        A single string (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
        or a callable (https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) to evaluate the
        predictions on the test set.
        If None, the estimator’s score method is used.
        Multimetric scoring is available for inner loops. If multimetric, outer loop will only be score with metric
        specified by refit_inner
        See sklearn.model_selection.GridSearchCV for more details.
    verbose: int (default=1)
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric (default=np.nan)
        Value to assign to the score if an error occurs in estimator fitting. If set to ‘raise’, the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter does not affect the refit step, which
        will always raise the error.
    refit_inner: boolean, string or callable (default=True)
        Refit an estimator using the best found parameters on the whole outer training set
        Argument will be given to search method that select hyperparameters in the inner loop :
            For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the
            best parameters for refitting the estimator at the end. Where there are considerations other than maximum
            score in choosing a best estimator, refit can be set to a function which returns the selected best_index_
            given cv_results_. In that case, the best_estimator_ and best_parameters_ will be set according to the
            returned best_index_.The refitted estimator is made available at the best_estimator_ attribute and permits
            using predict directly on this RandomSearchCV instance. If inner loops use multimetric scoring and the user
            want to refit according to a callable, the latter must be passed in a dictionary {score: callable} with
            score being the score name with which the score on different sets will be calculated
    refit_outer: boolean (default=True)
        Refit an estimator using the whole dataset in two steps:
        1. Hyperparameter optimization with a random search cross-validation (same parameter as outer CV).
        2. Refit an estimator using the best found parameters on the whole dataset.
    return_train_score: boolean (default=False)
        If False, the cross_validation results attribute will not include training scores.
        Computing training scores is used to get insights on how different parameter settings impact
        the overfitting/underfitting trade-off. However computing the scores on the training set can be
        computationally expensive and is not strictly required to select the parameters that yield the best
        generalization performance.
    n_iter: int (default=10)
        Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    random_state: int, RandomState instance or None (default=None)
        Pseudo random number generator state used for random uniform sampling from lists of possible values instead
        of scipy.stats distributions. Pass an int for reproducible output across multiple function calls.
    """

    def __init__(
        self,
        pipeline_dic,
        params_dic,
        outer_cv=5,
        inner_cv=5,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        imblearn_pipeline=False,
        pipeline_options=None,
        metric="roc_auc",
        verbose=1,
        refit_outer=True,
        error_score=np.nan,
        refit_inner=True,
        return_train_score=False,
        n_iter=10,
        random_state=None,
    ):
        super().__init__(
            pipeline_dic,
            params_dic,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            imblearn_pipeline=imblearn_pipeline,
            pipeline_options=pipeline_options,
            metric=metric,
            verbose=verbose,
            refit_outer=refit_outer,
            error_score=error_score,
            refit_inner=refit_inner,
            return_train_score=return_train_score,
        )
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_inner_param_optimizer(self, inner_model, inner_cv):
        return RandomizedSearchCV(
            inner_model,
            self.params_grid,
            scoring=self.scorers,
            n_jobs=self.n_jobs,
            cv=inner_cv,
            n_iter=self.n_iter,
            return_train_score=self.return_train_score,
            verbose=self.verbose - 1,
            pre_dispatch=self.pre_dispatch,
            refit=self.refit_inner,
            random_state=self.random_state,
            error_score=self.error_score,
        )

    def _get_outer_param_optimizer(self, final_model, outer_cv):
        return RandomizedSearchCV(
            final_model,
            self.params_grid,
            scoring=self.scorers[self.refit_metric],
            n_jobs=self.n_jobs,
            cv=outer_cv,
            n_iter=self.n_iter,
            return_train_score=self.return_train_score,
            verbose=self.verbose - 1,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            random_state=self.random_state,
        )
