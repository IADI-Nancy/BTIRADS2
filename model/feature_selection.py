"""'The nested cross-validation and feature selection pipeline are adapted from
 https://github.com/TimZaragori/Sklearn_NestedCV/tree/master"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numbers
import pandas as pd
from scipy.stats import kruskal, pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    SelectKBest,
    SelectFromModel,
    RFE,
    RFECV,
)
from sklearn.utils import resample
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from abc import abstractmethod
from sklearn.ensemble import ExtraTreesClassifier
from model.feature_selection_cv import CV
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt


def get_fs_dict(
    fs_metric,
    fs_nfeat_list,
    fs_aggregation,
    feature_key="FilterFeatureSelection",
    scaler=MinMaxScaler,
    use_scale=True,
    random_state=None,
    save_intermediate=False,
):
    fs_dict = {}

    for fs_nfeat in fs_nfeat_list:

        model_info = {"model_name": "FS"}

        pipeline_dict = {}
        params_dict = {}
        pipeline_options = {}

        if use_scale:
            pipeline_dict.update({"scale": scaler})

        if fs_aggregation is not None:
            print("Feature selection with bootstrap aggregation")
            pipeline_options.update(
                {
                    feature_key: {
                        "bootstrap": True,
                        "n_bsamples": 100,
                        "n_selected_features": fs_nfeat,
                        "ranking_aggregation": fs_aggregation,
                        "save_intermediate": save_intermediate,
                    }
                }
            )
        else:
            print("Simple feature selection without bootstrap aggregation")
            pipeline_options.update(
                {
                    feature_key: {
                        "bootstrap": False,
                        "n_selected_features": fs_nfeat,
                        "ranking_aggregation": fs_aggregation,
                        "save_intermediate": save_intermediate,
                    }
                }
            )

        fs_info = {
            "fs_metric": fs_metric,
            "fs_aggregation": fs_aggregation,
            "fs_nfeat": fs_nfeat,
        }
        name_suffix = "%s_%s_%i" % (fs_metric, fs_aggregation, fs_nfeat)

        pipeline_dict.update({feature_key: fs_metric})
        fs_dict.update(
            {
                "%s"
                % (name_suffix): {
                    "pipeline_dict": pipeline_dict,
                    "params_dict": params_dict,
                    "pipeline_options": pipeline_options,
                    "model_info": model_info,
                    "fs_info": fs_info,
                }
            }
        )

    return fs_dict


def perform_feature_selection(
    X, y, pipeline_dict, params_dict, pipeline_options={}, n_jobs=None, verbose=2
):

    # we use all train samples rather relying on repeated k-fold splits
    # for compatibility with cv pipelines, we pass a single split object
    cv = NoSplit()

    fs_pipeline = CV(
        pipeline_dict,
        params_dict,
        cv=cv,
        n_jobs=n_jobs,
        pipeline_options=pipeline_options,
        verbose=verbose,
    )
    fs_pipeline.fit(X, y)

    return fs_pipeline


def get_feature_mask_from_cv_results(X, n_selected_features, save_dir, verbose=False):

    # Load the feature selection model
    model_pickle_dir = os.path.join(save_dir, "Pickled_model")
    model_pkl_i = load(
        os.path.join(model_pickle_dir, "joblib_model_with_info_outer%s.pkl" % 0)
    )
    model_i = model_pkl_i["Model"]
    try:
        fs = model_i["FilterFeatureSelection"]
    except:
        fs = model_i["ModelFeatureSelection"]

    if not fs.bootstrap:
        scores, pvalues, ranks, ranking_index = fs._get_ranks()

        # pvalue_counts = pd.DataFrame({"p-values < 0.05": np.sum(pvalues < 0.05), "p-values < 0.1": np.sum(pvalues < 0.1), "p-values < 0.2": np.sum(pvalues < 0.2)}, index=[0])
        # pvalue_counts.to_excel(os.path.join(save_dir,  'pvalue_counts.xlsx'))

        # Get the feature mask for the feature ranking
        sorted_feature_indices = np.argsort(ranks)
        feature_mask = np.zeros(X.shape[1], dtype=bool)
        feature_mask[sorted_feature_indices[:n_selected_features]] = True

        if verbose:
            ranking_data_list = [scores, pvalues, ranks]
            ranking_name_list = ["scores", "pvalues", "ranks"]

            for ranking_data, ranking_name in zip(ranking_data_list, ranking_name_list):

                fig, axes = plt.subplots(figsize=(45, 25))
                sns.boxplot(
                    data=[ranking_data[None, f] for f in sorted_feature_indices],
                    ax=axes,
                )
                axes.set_xlabel("Feature", fontsize=24)
                axes.set_ylabel(ranking_name, fontsize=24)
                axes.minorticks_on()
                axes.xaxis.set_tick_params(which="minor", bottom=False)

                plt.setp(
                    axes,
                    xticks=range(len(sorted_feature_indices)),
                    xticklabels=[X.columns.values[f] for f in sorted_feature_indices],
                )
                plt.setp(axes.xaxis.get_majorticklabels(), rotation=90)
                plt.setp(axes.get_xticklabels(), fontsize=20)
                plt.setp(axes.get_yticklabels(), fontsize=20)

                if ranking_name == "pvalues":
                    plt.grid(True, which="both")

                plt.savefig(
                    os.path.join(save_dir, ranking_name + "_boxplot.png"),
                    bbox_inches="tight",
                )
                plt.close()

                ranking_df = pd.DataFrame(
                    {
                        "Feature": [
                            X.columns.values[f] for f in sorted_feature_indices
                        ],
                        "value": np.round(ranking_data[sorted_feature_indices], 4),
                        "n nan": np.count_nonzero(
                            np.isnan(ranking_data[sorted_feature_indices]), axis=0
                        ),
                    }
                )
                ranking_df.to_excel(
                    os.path.join(save_dir, ranking_name + "_stats.xlsx")
                )

    else:
        assert fs.ranking_aggregation is not None, "currently not supported"

        # Get the bootstrap results (ranking index = index of the best selected features,  bootstrap_scores_aggregated = rank for each feature)
        (
            bootstrap_scores,
            bootstrap_pvalues,
            bootstrap_ranks,
            bootstrap_scores_aggregated,
            bootstrap_ranks_aggregated,
            ranking_index,
        ) = fs._get_bootstrap_ranks()

        # ranking_index: list indexing best to worst N features
        # bootstrap_ranks_aggregated: [N]
        # bootstrap_pvalues: [100xN] (100 bootstrap samples)
        # pvalue_counts = pd.DataFrame({"median p-values < 0.05": np.sum(np.nanmedian(bootstrap_pvalues, axis=0) < 0.05), "median p-values < 0.1": np.sum(np.nanmedian(bootstrap_pvalues, axis=0) < 0.1), "median p-values < 0.2": np.sum(np.nanmedian(bootstrap_pvalues, axis=0) < 0.2)}, index=[0])
        # pvalue_counts.to_excel(os.path.join(save_dir,  'pvalue_counts.xlsx'))

        # Each feature was assigned a rank (bootstrap_ranks_aggregated), select the n_selected_features with best(lowest) rank
        # Get the feature indices that sort the ranking array
        sorted_feature_indices = np.argsort(bootstrap_ranks_aggregated)
        feature_mask = np.zeros(X.shape[1], dtype=bool)
        feature_mask[sorted_feature_indices[:n_selected_features]] = True

        if verbose:
            ranking_data_list = [
                bootstrap_scores,
                bootstrap_pvalues,
                bootstrap_ranks,
                bootstrap_ranks_aggregated,
            ]  # bootstrap_scores_aggregated,
            ranking_name_list = [
                "bootstrap_scores",
                "bootstrap_pvalues",
                "bootstrap_ranks",
                "bootstrap_aggregated_ranks",
            ]  #'bootstrap_aggregated_scores',

            for ranking_data, ranking_name in zip(ranking_data_list, ranking_name_list):

                # If data has shape (n_bootstrap,n_features), compute the median over the bootstrap samples (axis=0) -> output shape (n_features)
                if ranking_data.ndim == 2:
                    reduced_ranking_data = np.nanmedian(ranking_data, axis=0)
                else:
                    reduced_ranking_data = ranking_data
                    ranking_data = ranking_data[None, :]  # unsqueeze to (1,n_features)

                fig, axes = plt.subplots(figsize=(45, 25))
                sns.boxplot(
                    data=[ranking_data[:, f] for f in sorted_feature_indices], ax=axes
                )
                axes.set_xlabel("Feature", fontsize=24)
                axes.set_ylabel(ranking_name, fontsize=24)
                axes.minorticks_on()
                axes.xaxis.set_tick_params(which="minor", bottom=False)
                plt.setp(
                    axes,
                    xticks=range(len(sorted_feature_indices)),
                    xticklabels=[X.columns.values[f] for f in sorted_feature_indices],
                )
                plt.setp(axes.xaxis.get_majorticklabels(), rotation=90)
                plt.setp(axes.get_xticklabels(), fontsize=20)
                plt.setp(axes.get_yticklabels(), fontsize=20)

                if ranking_name == "bootstrap_pvalues":
                    plt.grid(True, which="both")

                plt.savefig(
                    os.path.join(save_dir, ranking_name + "_boxplot.png"),
                    bbox_inches="tight",
                )  # , dpi=600
                plt.close()

                ranking_df = pd.DataFrame(
                    {
                        "Feature": [
                            X.columns.values[f] for f in sorted_feature_indices
                        ],
                        "mean": np.round(
                            np.nanmean(ranking_data[:, sorted_feature_indices], axis=0),
                            4,
                        ),
                        "std": np.round(
                            np.nanstd(ranking_data[:, sorted_feature_indices], axis=0),
                            4,
                        ),
                        "25th percentile": np.round(
                            np.nanpercentile(
                                ranking_data[:, sorted_feature_indices], 25, axis=0
                            ),
                            4,
                        ),
                        "50th percentile": np.round(
                            np.nanpercentile(
                                ranking_data[:, sorted_feature_indices], 50, axis=0
                            ),
                            4,
                        ),
                        "75th percentile": np.round(
                            np.nanpercentile(
                                ranking_data[:, sorted_feature_indices], 75, axis=0
                            ),
                            4,
                        ),
                        "n nan": np.count_nonzero(
                            np.isnan(ranking_data[:, sorted_feature_indices]), axis=0
                        ),
                    }
                )
                ranking_df.to_excel(
                    os.path.join(save_dir, ranking_name + "_stats.xlsx")
                )

    # This is uggly...
    if "Expandability" in "\t".join(X.iloc[:, feature_mask].columns.values):
        print(
            "Adding Expandability_not applicable to the selected feature list to enable distinguishing patients without follow-up scans"
        )
        feature_mask[
            [
                f
                for f in sorted_feature_indices
                if X.columns.values[f] == "Expandability_not applicable"
            ]
        ] = True

    return feature_mask


class NoSplit:

    def split(self, X, y):
        for i in range(1):
            train_idx = np.arange(X.shape[0])
            yield train_idx, train_idx

    def get_n_splits(self):
        return 1


class FeatureSelection(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    Abstract class for feature selection
    """

    @staticmethod
    def _check_X_Y(X, y=None):
        # Check X
        if not isinstance(X, (list, tuple, np.ndarray)):
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X = X.to_numpy()
            else:
                raise TypeError(
                    "X array must be an array like or pandas Dataframe/Series"
                )
        else:
            X = np.array(X)
        if len(X.shape) != 2:
            raise ValueError("X array must 2D")
        if y is not None:
            # Check y
            if not isinstance(y, (list, tuple, np.ndarray)):
                if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                    y = y.to_numpy()
                else:
                    raise TypeError(
                        "y array must be an array like or pandas Dataframe/Series"
                    )
            else:
                y = np.array(y)
            if len(y.shape) != 1:
                if len(y.shape) == 2 and y.shape[1] == 1:
                    y.reshape(-1)
                else:
                    raise ValueError(
                        "y array must be 1D or 2D with second dimension equal to 1"
                    )
            if len(np.unique(y)) <= 1:
                raise ValueError("y array must have at least 2 classes")
        return X, y

    @abstractmethod
    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

    def transform(self, X):
        """Select the n_selected_features best features to create a new dataset.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        n_selected_features
             array of shape (n_samples, n_selected_features) containing the selected features
        """

        X, _ = self._check_X_Y(X, None)

        check_is_fitted(self)
        self.selected_features = super(FeatureSelection, self).transform(X)
        return self.selected_features

    def fit_transform(self, X, y=None, **fit_params):
        """A method to fit feature selection and reduce X to selected features.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        n_selected_features
            array of shape (n_samples, n_selected_features) containing the selected features
        You should be able to access as class attribute to:
        ranking_index
             A list of features indexes sorted by ranks. ranking_index[0] returns the index of the best selected
             feature according to scoring/ranking function
        """
        return self.fit(X, y, **fit_params).transform(X)


class FilterFeatureSelection(FeatureSelection):
    """A general class to handle feature selection according to a scoring/ranking method. Bootstrap is implemented
    to ensure stability in feature selection process
    Parameters
    ----------
    method: str or callable
        Method used to score/rank features. Either str or callable
        If str inbuild function named as str is called, must be one of following:
            'wlcx_score': score of kruskall wallis test
            'auc_roc': scoring with area under the roc curve
            'pearson_corr': scoring with pearson correlation coefficient between features and labels
            'spearman_corr': scoring with spearman correlation coefficient between features and labels
            'mi': scoring with mutual information between features and labels
            'mrmr': ranking according to Minimum redundancy Maximum relevance algorithm
        If callable method must take (feature_array,label_array) as arguments and return either the score or the rank
        associated with each feature in the same order as features are in feature_array
    bootstrap: boolean (default=False)
        Choose whether feature selection must be done in a bootstraped way
    n_bsamples: int (default=100)
        Number of bootstrap samples generated for bootstraped feature selection. Ignored if bootstrap is False
    n_selected_features: int or None, default = 20
        Number of the best features that must be selected by the class
        If None all the feature are returned (no feature selection)
    ranking_aggregation: str or callable (default=None)
        Method used to aggregate rank of bootstrap samples. Either str or callable
        If str inbuild function named as str is called, must be one of following:'enhanced_borda', 'borda',
        'importance_score', 'mean', 'stability_selection', 'exponential_weighting'
        If callable method must take ((bootstrap_ranks, n_selected_features) as arguments and return the
        aggregate rank associated with each feature in the same order as features are in feature_array
    ranking_done: boolean (default=False)
        Indicate whether the method return a score or directly calculate ranks
    score_indicator_lower: boolean (default=None)
        Choose whether lower score correspond to higher rank for the rank calculation or higher score is better,
        `True` means lower score is better. Determined automatically for inbuild functions
    classification: boolean (default=True)
        Define whether the current problem is a classification problem.
    random_state: int, RandomState instance or None (default=None)
        Controls the randomness of the estimator.
    """

    scoring_methods = {
        "name": [
            "auc_roc",
            "pearson_corr",
            "spearman_corr",
            "mi",
            "wlcx_score",
            "chi2",
        ],
        "score_indicator_lower": [False, False, False, False, False, False],
    }
    ranking_methods = ["mrmr"]
    ranking_aggregation_methods = [
        "enhanced_borda",
        "borda",
        "importance_score",
        "mean",
        "median",
        "stability_selection_aggregation",
        "exponential_weighting",
    ]

    def __init__(
        self,
        method="mrmr",
        bootstrap=False,
        n_bsamples=100,
        n_selected_features=20,
        ranking_aggregation=None,
        ranking_done=False,
        score_indicator_lower=None,
        classification=True,
        random_state=None,
        save_intermediate=False,
    ):
        self.method = method
        self.ranking_done = ranking_done
        self.score_indicator_lower = score_indicator_lower

        self.bootstrap = bootstrap
        self.n_bsamples = n_bsamples
        self.n_selected_features = n_selected_features
        self.ranking_aggregation = ranking_aggregation
        self.classification = classification
        self.random_state = random_state
        self.save_intermediate = save_intermediate

    def _get_fs_func(self):
        if callable(self.method):
            return self.method
        elif isinstance(self.method, str):
            method_name = self.method.lower()
            if method_name not in (self.scoring_methods["name"] + self.ranking_methods):
                raise ValueError(
                    "If string method must be one of : %s. "
                    "%s was passed"
                    % (
                        str(self.scoring_methods["name"] + self.ranking_methods),
                        method_name,
                    )
                )
            if method_name in self.ranking_methods:
                self.ranking_done = True
            elif method_name in self.scoring_methods["name"]:
                self.ranking_done = False
                self.score_indicator_lower = self.scoring_methods[
                    "score_indicator_lower"
                ][self.scoring_methods["name"].index(self.method)]
            else:
                raise ValueError(
                    "If string method must be one of : %s. "
                    "%s was passed"
                    % (
                        str(self.scoring_methods["name"] + self.ranking_methods),
                        method_name,
                    )
                )
            return getattr(self, method_name)
        else:
            raise TypeError("method argument must be a callable or a string")

    def _get_aggregation_method(self):
        if not callable(self.ranking_aggregation) and not isinstance(
            self.ranking_aggregation, str
        ):
            raise TypeError("ranking_aggregation option must be a callable or a string")
        else:
            if isinstance(self.ranking_aggregation, str):
                ranking_aggregation_name = self.ranking_aggregation.lower()
                if self.ranking_aggregation not in self.ranking_aggregation_methods:
                    raise ValueError(
                        "If string ranking_aggregation must be one of : {0}. "
                        "%s was passed".format(
                            str(self.ranking_aggregation_methods),
                            ranking_aggregation_name,
                        )
                    )
                return getattr(FilterFeatureSelection, self.ranking_aggregation)

    @staticmethod
    def _check_n_selected_feature(X, n_selected_features):
        if (
            not isinstance(n_selected_features, numbers.Integral)
            and n_selected_features is not None
        ):
            raise TypeError("n_selected_feature must be int or None")
        else:
            if n_selected_features is None:
                n_selected_features = X.shape[1]
            else:
                n_selected_features = n_selected_features
            return n_selected_features

    def _get_bsamples_index(self, y):
        bsamples_index = []
        n = 0
        while len(bsamples_index) < self.n_bsamples:
            bootstrap_sample = resample(range(self.n_samples), random_state=n)
            # Ensure all classes are present in bootstrap sample.
            if len(np.unique(y[bootstrap_sample])) == self.n_classes:
                bsamples_index.append(bootstrap_sample)
            n += 1
        bsamples_index = np.array(bsamples_index)
        return bsamples_index

    def _get_support_mask(self):
        mask = np.zeros(self.n_features, dtype=bool)
        mask[self.accepted_features_index_] = True
        return mask

    # === Scoring methods ===
    @staticmethod
    def wlcx_score(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        labels = np.unique(y)
        for i in range(n_features):
            X_by_label = [X[:, i][y == _] for _ in labels]
            statistic, pvalue = kruskal(*X_by_label)
            score[i] = statistic
        return score

    @staticmethod
    def auc_roc(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        labels = np.unique(y)
        if len(labels) == 2:
            for i in range(n_features):
                # Replicate of roc function from pROC R package to find positive class
                control_median = np.median(X[:, i][y == labels[0]])
                case_median = np.median(X[:, i][y == labels[1]])
                if case_median > control_median:
                    positive_label = 1
                else:
                    positive_label = 0
                fpr, tpr, thresholds = roc_curve(y, X[:, i], pos_label=positive_label)
                roc_auc = auc(fpr, tpr)
                score[i] = roc_auc
        else:
            # Adapted from roc_auc_score for multi_class labels
            # See sklearn.metrics._base_average_multiclass_ovo_score
            # Hand & Till (2001) implementation (ovo)
            n_classes = labels.shape[0]
            n_pairs = n_classes * (n_classes - 1) // 2
            for i in range(n_features):
                pair_scores = np.empty(n_pairs)
                for ix, (a, b) in enumerate(combinations(labels, 2)):
                    a_mask = y == a
                    b_mask = y == b
                    ab_mask = np.logical_or(a_mask, b_mask)

                    # Replicate of roc function from pROC R package to find positive class
                    control_median = np.median(X[:, i][ab_mask][y[ab_mask] == a])
                    case_median = np.median(X[:, i][ab_mask][y[ab_mask] == b])
                    if control_median > case_median:
                        positive_class = y[ab_mask] == a
                    else:
                        positive_class = y[ab_mask] == b
                    fpr, tpr, _ = roc_curve(positive_class, X[:, i][ab_mask])
                    roc_auc = auc(fpr, tpr)
                    pair_scores[ix] = roc_auc
                score[i] = np.average(pair_scores)
        return score

    @staticmethod
    def pearson_corr(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            correlation, pvalue = pearsonr(X[:, i], y)
            score[i] = correlation
        return score

    @staticmethod
    def spearman_corr(X, y):
        n_samples, n_features = X.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            correlation, pvalue = spearmanr(X[:, i], y)
            score[i] = correlation
        return score

    def mi(self, X, y):
        score = mutual_info_classif(X, y, random_state=self.random_state)
        return score

    def chi2(self, X, y):
        assert (
            np.sum(X < 0) == 0
        ), "Chi2 does not support negative inputs, use a different data scaler (e.g. MinMaxScaler)!"
        selector = SelectKBest(chi2, k="all")
        selector.fit_transform(X, y)
        score = selector.scores_
        pvalue = selector.pvalues_

        return score, pvalue

    # === Ranking methods ===
    # @staticmethod
    # def mrmr(X, y):
    #     n_samples, n_features = X.shape
    #     rank_index, _, _ = mrmr(X, y, n_selected_features=n_features)
    #     ranks = np.array([list(rank_index).index(_) + 1 for _ in range(len(rank_index))])
    #     return ranks

    # === Ranking aggregation methods ===
    # Importance score method :
    # A comparative study of machine learning methods for time-to-event survival data for radiomics risk modelling
    # Leger et al., 2017, Scientific Reports
    # Other methods:
    # An extensive comparison of feature ranking aggregation techniques in bioinformatics.
    # Randall et al., 2012, IEEE
    @staticmethod
    def borda(bootstrap_ranks, n_selected_features):
        return np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0) * -1

    @staticmethod
    def mean(bootstrap_ranks, n_selected_features):
        return np.mean(bootstrap_ranks, axis=0)

    @staticmethod
    def median(bootstrap_ranks, n_selected_features):
        return np.median(bootstrap_ranks, axis=0)

    @staticmethod
    def stability_selection_aggregation(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return np.sum(bootstrap_ranks <= n_selected_features, axis=0) * -1

    @staticmethod
    def exponential_weighting(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return np.sum(np.exp(-bootstrap_ranks / n_selected_features), axis=0) * -1

    @staticmethod
    def enhanced_borda(bootstrap_ranks, n_selected_features):
        borda_count = np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0)
        stability_selection = np.sum(bootstrap_ranks <= n_selected_features, axis=0)
        return borda_count * stability_selection * -1

    @staticmethod
    def importance_score(bootstrap_ranks, n_selected_features):
        """
        A comparative study of machine learning methods for time-to-event survival data for
        radiomics risk modelling. Leger et al., Scientific Reports, 2017
        """
        occurence = np.sum(bootstrap_ranks <= n_selected_features, axis=0) ** 2
        importance_score = np.divide(
            np.sum(np.sqrt(bootstrap_ranks), axis=0),
            occurence,
            out=np.full(occurence.shape, np.inf),
            where=occurence != 0,
        )
        return importance_score

    # === Applying feature selection ===
    def fit(self, X, y=None):
        """
        A method to fit feature selection.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        Instance of fitted estimator.
        """
        X, y = self._check_X_Y(X, y)
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        fs_func = self._get_fs_func()
        if self.ranking_aggregation is not None:
            aggregation_method = self._get_aggregation_method()
        self.n_selected_features = self._check_n_selected_feature(
            X, self.n_selected_features
        )
        if self.bootstrap:
            if self.ranking_aggregation is None:
                raise ValueError(
                    "ranking_aggregation option must be given if bootstrap is True"
                )
            bsamples_index = self._get_bsamples_index(y)
            if self.ranking_done:
                bootstrap_ranks = np.array(
                    [fs_func(X[_, :], y[_]) for _ in bsamples_index]
                )
            else:
                if self.score_indicator_lower is None:
                    raise ValueError(
                        "score_indicator_lower option must be given if a user scoring function is used"
                    )
                bootstrap_scores = np.array(
                    [
                        (
                            fs_func(X[_, :], y[_])[0]
                            if isinstance(fs_func(X[_, :], y[_]), tuple)
                            else fs_func(X[_, :], y[_])
                        )
                        for _ in bsamples_index
                    ]
                )
                bootstrap_pvalues = np.array(
                    [
                        fs_func(X[_, :], y[_])[1]
                        for _ in bsamples_index
                        if isinstance(fs_func(X[_, :], y[_]), tuple)
                    ]
                )

                if not self.score_indicator_lower:
                    bootstrap_scores *= -1

                bootstrap_ranks = np.array([rankdata(_) for _ in bootstrap_scores])

            bootstrap_scores_aggregated = aggregation_method(
                bootstrap_ranks, self.n_selected_features
            )
            bootstrap_ranks_aggregated = rankdata(
                bootstrap_scores_aggregated, method="ordinal"
            )

            ranking_index = [
                list(bootstrap_ranks_aggregated).index(_)
                for _ in sorted(bootstrap_ranks_aggregated)
            ]
            if self.save_intermediate:
                if self.ranking_done:
                    print("DEBUG ranking done!")
                    self.bootstrap_scores = bootstrap_ranks
                    self.bootstrap_pvalues = []
                else:
                    self.bootstrap_scores = np.array(bootstrap_scores)
                    self.bootstrap_pvalues = bootstrap_pvalues
                self.bootstrap_ranks = bootstrap_ranks
                self.bootstrap_scores_aggregated = bootstrap_scores_aggregated
                self.bootstrap_ranks_aggregated = bootstrap_ranks_aggregated

                self.ranking_index = ranking_index

        else:
            if self.ranking_done:
                ranks = fs_func(X, y)
            else:
                if self.score_indicator_lower is None:
                    raise ValueError(
                        "score_indicator_lower option must be given if a user scoring function is used"
                    )
                score = (
                    fs_func(X, y)[0]
                    if isinstance(fs_func(X, y), tuple)
                    else fs_func(X, y)
                )
                pvalue = fs_func(X, y)[1] if isinstance(fs_func(X, y), tuple) else None

                if not self.score_indicator_lower:
                    score *= -1

                ranks = rankdata(score, method="ordinal")
            ranking_index = [list(ranks).index(_) for _ in sorted(ranks)]

            if self.save_intermediate:
                self.score = score
                self.pvalue = pvalue
                self.ranks = ranks
                self.ranking_index = ranking_index

        self.accepted_features_index_ = ranking_index[: self.n_selected_features]
        return self

    def _get_bootstrap_ranks(self):
        return (
            self.bootstrap_scores,
            self.bootstrap_pvalues,
            self.bootstrap_ranks,
            self.bootstrap_scores_aggregated,
            self.bootstrap_ranks_aggregated,
            self.ranking_index,
        )

    def _get_ranks(self):
        return self.score, self.pvalue, self.ranks, self.ranking_index


class ModelFeatureSelection(FeatureSelection):
    """A general class to handle feature selection according to a model_based scoring/ranking method based. Bootstrap is implemented
    to ensure stability in feature selection process
    Parameters
    ----------
    method: str or callable
        Method used to score/rank features. Either str or callable
        If str inbuild function named as str is called, must be one of following:
            'SelectFromModel':
        If callable method must take (feature_array,label_array) as arguments and return either the score or the rank
        associated with each feature in the same order as features are in feature_array

    bootstrap: boolean (default=False)
        Choose whether feature selection must be done in a bootstraped way
    n_bsamples: int (default=100)
        Number of bootstrap samples generated for bootstraped feature selection. Ignored if bootstrap is False
    n_selected_features: int or None, default = 20
        Number of the best features that must be selected by the class
        If None all the feature are returned (no feature selection)
    ranking_aggregation: str or callable (default=None)
        Method used to aggregate rank of bootstrap samples. Either str or callable
        If str inbuild function named as str is called, must be one of following:'enhanced_borda', 'borda',
        'importance_score', 'mean', 'stability_selection', 'exponential_weighting'
        If callable method must take ((bootstrap_ranks, n_selected_features) as arguments and return the
        aggregate rank associated with each feature in the same order as features are in feature_array
    ranking_done: boolean (default=False)
        Indicate whether the method return a score or directly calculate ranks
    score_indicator_lower: boolean (default=None)
        Choose whether lower score correspond to higher rank for the rank calculation or higher score is better,
        `True` means lower score is better. Determined automatically for inbuild functions
    classification: boolean (default=True)
        Define whether the current problem is a classification problem.
    random_state: int, RandomState instance or None (default=None)
        Controls the randomness of the estimator.
    """

    scoring_methods = {"name": ["select_from_model"], "score_indicator_lower": [False]}
    ranking_methods = ["rfe", "rfecv"]

    ranking_aggregation_methods = [
        "enhanced_borda",
        "borda",
        "importance_score",
        "mean",
        "median",
        "stability_selection_aggregation",
        "exponential_weighting",
    ]

    def __init__(
        self,
        method="select_from_model",
        estimator=None,
        bootstrap=False,
        n_bsamples=100,
        n_selected_features=20,
        ranking_aggregation=None,
        ranking_done=False,
        score_indicator_lower=None,
        classification=True,
        random_state=None,
        save_intermediate=False,
    ):
        self.method = method
        if estimator is not None:
            self.estimator = estimator
        else:
            # self.estimator = LogisticRegression()
            self.estimator = ExtraTreesClassifier(n_estimators=100)
        self.ranking_done = ranking_done
        self.score_indicator_lower = score_indicator_lower
        self.bootstrap = bootstrap
        self.n_bsamples = n_bsamples
        self.n_selected_features = n_selected_features
        self.ranking_aggregation = ranking_aggregation
        self.classification = classification
        self.random_state = random_state
        self.save_intermediate = save_intermediate

    def _get_fs_func(self):
        if callable(self.method):
            return self.method
        elif isinstance(self.method, str):
            method_name = self.method.lower()
            if method_name not in (self.scoring_methods["name"] + self.ranking_methods):
                raise ValueError(
                    "If string method must be one of : %s. "
                    "%s was passed"
                    % (
                        str(self.scoring_methods["name"] + self.ranking_methods),
                        method_name,
                    )
                )
            if method_name in self.ranking_methods:
                self.ranking_done = True
            elif method_name in self.scoring_methods["name"]:
                self.ranking_done = False
                self.score_indicator_lower = self.scoring_methods[
                    "score_indicator_lower"
                ][self.scoring_methods["name"].index(self.method)]
            else:
                raise ValueError(
                    "If string method must be one of : %s. "
                    "%s was passed"
                    % (
                        str(self.scoring_methods["name"] + self.ranking_methods),
                        method_name,
                    )
                )
            return getattr(self, method_name)
        else:
            raise TypeError("method argument must be a callable or a string")

    def _get_aggregation_method(self):
        if not callable(self.ranking_aggregation) and not isinstance(
            self.ranking_aggregation, str
        ):
            raise TypeError("ranking_aggregation option must be a callable or a string")
        else:
            if isinstance(self.ranking_aggregation, str):
                ranking_aggregation_name = self.ranking_aggregation.lower()
                if self.ranking_aggregation not in self.ranking_aggregation_methods:
                    raise ValueError(
                        "If string ranking_aggregation must be one of : {0}. "
                        "%s was passed".format(
                            str(self.ranking_aggregation_methods),
                            ranking_aggregation_name,
                        )
                    )
                return getattr(FilterFeatureSelection, self.ranking_aggregation)

    @staticmethod
    def _check_n_selected_feature(X, n_selected_features):
        if (
            not isinstance(n_selected_features, numbers.Integral)
            and n_selected_features is not None
        ):
            raise TypeError("n_selected_feature must be int or None")
        else:
            if n_selected_features is None:
                n_selected_features = X.shape[1]
            else:
                n_selected_features = n_selected_features
            return n_selected_features

    def _get_bsamples_index(self, y):
        bsamples_index = []
        n = 0
        while len(bsamples_index) < self.n_bsamples:
            bootstrap_sample = resample(range(self.n_samples), random_state=n)
            # Ensure all classes are present in bootstrap sample.
            if len(np.unique(y[bootstrap_sample])) == self.n_classes:
                bsamples_index.append(bootstrap_sample)
            n += 1
        bsamples_index = np.array(bsamples_index)
        return bsamples_index

    def _get_support_mask(self):
        mask = np.zeros(self.n_features, dtype=bool)
        mask[self.accepted_features_index_] = True
        return mask

    # === Importance extraction methods ===
    @staticmethod
    def select_from_model(estimator, X, y):
        selector = SelectFromModel(estimator=estimator).fit(X, y)

        if hasattr(selector.estimator_, "feature_importances_"):
            score = selector.estimator_.feature_importances_
        elif hasattr(selector.estimator_, "coef_"):
            score = np.abs(selector.estimator_.coef_)  # importance value
        else:
            raise TypeError(
                "estimator unsuitable for SelectFromModel (no feature_importances_/coef_ attribute)"
            )

        return score

    def rfe(estimator, X, y):
        selector = RFE(estimator=estimator).fit(X, y)
        ranking = selector.ranking_

        return ranking

    def rfecv(estimator, X, y):
        selector = RFECV(estimator=estimator).fit(X, y)  # default is 5-fold crossval
        ranking = selector.ranking_

        return ranking

    # === Ranking aggregation methods ===
    # Importance score method :
    # A comparative study of machine learning methods for time-to-event survival data for radiomics risk modelling
    # Leger et al., 2017, Scientific Reports
    # Other methods:
    # An extensive comparison of feature ranking aggregation techniques in bioinformatics.
    # Randall et al., 2012, IEEE
    @staticmethod
    def borda(bootstrap_ranks, n_selected_features):
        return np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0) * -1

    @staticmethod
    def mean(bootstrap_ranks, n_selected_features):
        return np.mean(bootstrap_ranks, axis=0)

    @staticmethod
    def median(bootstrap_ranks, n_selected_features):
        return np.median(bootstrap_ranks, axis=0)

    @staticmethod
    def stability_selection_aggregation(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return np.sum(bootstrap_ranks <= n_selected_features, axis=0) * -1

    @staticmethod
    def exponential_weighting(bootstrap_ranks, n_selected_features):
        """
        A.-C. Haury, P. Gestraud, and J.-P. Vert,
        The influence of feature selection methods on accuracy, stability and interpretability of molecular signatures
        PLoS ONE
        """
        return np.sum(np.exp(-bootstrap_ranks / n_selected_features), axis=0) * -1

    @staticmethod
    def enhanced_borda(bootstrap_ranks, n_selected_features):
        borda_count = np.sum(bootstrap_ranks.shape[1] - bootstrap_ranks, axis=0)
        stability_selection = np.sum(bootstrap_ranks <= n_selected_features, axis=0)
        return borda_count * stability_selection * -1

    @staticmethod
    def importance_score(bootstrap_ranks, n_selected_features):
        """
        A comparative study of machine learning methods for time-to-event survival data for
        radiomics risk modelling. Leger et al., Scientific Reports, 2017
        """
        occurence = np.sum(bootstrap_ranks <= n_selected_features, axis=0) ** 2
        importance_score = np.divide(
            np.sum(np.sqrt(bootstrap_ranks), axis=0),
            occurence,
            out=np.full(occurence.shape, np.inf),
            where=occurence != 0,
        )
        return importance_score

    # === Applying feature selection ===
    def fit(self, X, y=None):
        """
        A method to fit feature selection.
        Parameters
        ----------
        X : pandas dataframe or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : pandas dataframe or array-like of shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
        Instance of fitted estimator.
        """
        X, y = self._check_X_Y(X, y)
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        fs_func = self._get_fs_func()
        if self.ranking_aggregation is not None:
            aggregation_method = self._get_aggregation_method()
        self.n_selected_features = self._check_n_selected_feature(
            X, self.n_selected_features
        )
        if self.bootstrap:
            if self.ranking_aggregation is None:
                raise ValueError(
                    "ranking_aggregation option must be given if bootstrap is True"
                )
            bsamples_index = self._get_bsamples_index(y)
            if self.ranking_done:
                bootstrap_ranks = np.array(
                    [fs_func(self.estimator, X[_, :], y[_]) for _ in bsamples_index]
                )
            else:
                if self.score_indicator_lower is None:
                    raise ValueError(
                        "score_indicator_lower option must be given if a user scoring function is used"
                    )
                bootstrap_scores = np.array(
                    [fs_func(self.estimator, X[_, :], y[_]) for _ in bsamples_index]
                )
                if not self.score_indicator_lower:
                    bootstrap_scores *= -1
                bootstrap_ranks = np.array([rankdata(_) for _ in bootstrap_scores])

            bootstrap_scores_aggregated = aggregation_method(
                bootstrap_ranks, self.n_selected_features
            )
            bootstrap_ranks_aggregated = rankdata(
                bootstrap_scores_aggregated, method="ordinal"
            )

            ranking_index = [
                list(bootstrap_ranks_aggregated).index(_)
                for _ in sorted(bootstrap_ranks_aggregated)
            ]

            if self.save_intermediate:
                if self.ranking_done:
                    self.bootstrap_scores = bootstrap_ranks
                else:
                    self.bootstrap_scores = np.array(bootstrap_scores)
                self.bootstrap_ranks = bootstrap_ranks
                self.bootstrap_scores_aggregated = bootstrap_scores_aggregated
                self.bootstrap_ranks_aggregated = bootstrap_ranks_aggregated
                self.ranking_index = ranking_index
        else:
            if self.ranking_done:
                ranks = fs_func(self.estimator, X, y)
            else:
                if self.score_indicator_lower is None:
                    raise ValueError(
                        "score_indicator_lower option must be given if a user scoring function is used"
                    )
                score = fs_func(self.estimator, X, y)
                if not self.score_indicator_lower:
                    score *= -1
                ranks = rankdata(score, method="ordinal")
            ranking_index = [list(ranks).index(_) for _ in sorted(ranks)]
        self.accepted_features_index_ = ranking_index[: self.n_selected_features]

        return self

    def _get_bootstrap_ranks(self):
        return (
            self.bootstrap_scores,
            [],
            self.bootstrap_ranks,
            self.bootstrap_scores_aggregated,
            self.bootstrap_ranks_aggregated,
            self.ranking_index,
        )
