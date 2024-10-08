import os
import pandas as pd
from utils.data_io import load_data
from utils.preprocessing import preprocess_pipeline
from utils.visualization import probability_histplot, probability_scatterplot
import joblib
import numpy as np
import ast
import re
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import imblearn
from scipy import stats
import warnings
from commands.btirads2 import BTIRADS2

warnings.filterwarnings("ignore", message=".*fitted without feature names*")
warnings.filterwarnings(
    "ignore",
    message=".*Dropping invalid columns in DataFrameGroupBy.mean is deprecated*",
)


class Evaluate:
    def __init__(self, args) -> None:
        self.args = args

    def get_performance_scores(
        self,
        pred_proba,
        pred,
        y_true,
        confidence_interval=True,
        return_bootstraps=False,
    ):

        # confidence intervals using test set bootstrapping
        if confidence_interval:
            df_bootstrapped = pd.DataFrame()

            rng = np.random.RandomState(seed=self.args.random_seed)
            bidx = np.arange(pred.shape[0])

            for b in range(self.args.n_bootstrap):
                pred_bidx = rng.choice(bidx, size=bidx.shape[0], replace=True)

                if pred_proba is None:
                    scores = self.get_performance_scores(
                        None,
                        pred[pred_bidx],
                        y_true[pred_bidx],
                        confidence_interval=False,
                    )
                else:
                    scores = self.get_performance_scores(
                        pred_proba[pred_bidx],
                        pred[pred_bidx],
                        y_true[pred_bidx],
                        confidence_interval=False,
                    )

                df_bootstrapped = pd.concat(
                    [
                        df_bootstrapped.reset_index(drop=True),
                        pd.DataFrame(dict({"bidx": b}, **scores), index=[0]),
                    ],
                    axis=0,
                )

            bootstrap_mean_scores = df_bootstrapped.mean()
            bootstrap_lower_scores = df_bootstrapped.quantile(self.args.ci[0])
            bootstrap_upper_scores = df_bootstrapped.quantile(self.args.ci[1])

            for k in scores.keys():
                scores[k] = (
                    str(np.round(bootstrap_mean_scores[k], 2))
                    + " ["
                    + str(np.round(bootstrap_lower_scores[k], 2))
                    + ","
                    + str(np.round(bootstrap_upper_scores[k], 2))
                    + "]"
                )

            if not return_bootstraps:
                return scores
            else:
                return scores, df_bootstrapped

        if pred_proba is not None:
            roc_auc = roc_auc_score(y_true, pred_proba)
        else:
            roc_auc = np.nan

        f1 = f1_score(y_true, pred, average="binary", pos_label=1)
        accuracy = accuracy_score(y_true, pred)
        sensitivity = imblearn.metrics.sensitivity_score(
            y_true, pred, average="binary", pos_label=1
        )
        specificity = imblearn.metrics.specificity_score(
            y_true, pred, average="binary", pos_label=1
        )

        # additional numbers to compute nominators and denominators
        n_correct = np.sum(y_true == pred)
        n_tp = np.sum((y_true == pred) & (y_true == 1))
        n_tn = np.sum((y_true == pred) & (y_true == 0))
        n_fp = np.sum((y_true != pred) & (y_true == 0))
        n_fn = np.sum((y_true != pred) & (y_true == 1))
        n_p = np.sum(y_true == 1)
        n_n = np.sum(y_true == 0)
        f1_nom = 2 * n_tp
        f1_den = 2 * n_tp + n_fp + n_fn
        scores = {
            "f1_score": f1,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "n_correct": n_correct,
            "n_tp": n_tp,
            "n_tn": n_tn,
            "n_p": n_p,
            "n_n": n_n,
            "f1_nom": f1_nom,
            "f1_den": f1_den,
        }

        return scores

    def exec(self):

        if not os.path.exists(os.path.join(self.args.outputs_dir)):
            os.makedirs(os.path.join(self.args.outputs_dir))

        if self.args.evaluate_cv:
            print("Start nested cross-validation evaluation")
            print("Loading file: ", self.args.data_dir)

            metrics = ["f1_score", "accuracy", "sensitivity", "specificity", "roc_auc"]
            cv_eval_df = pd.DataFrame(
                columns=[
                    "model",
                    "Modality, Model",
                    "Feature selection",
                    "fold",
                    "split",
                ]
            )
            cv_eval_summary_df = pd.DataFrame(
                columns=["model", "Modality, Model", "Feature selection", "split"]
                + [
                    m + ext
                    for m in metrics
                    for ext in [" mean", " std", " lower", " upper"]
                ]
            )

            for modality, modality_txt in zip(
                self.args.modalities, self.args.modality_txts
            ):

                mod_outputs_dir = os.path.join(self.args.outputs_dir, modality)
                summary_outputs_dir = os.path.dirname(mod_outputs_dir)

                # load the data
                data, label = load_data(self.args.data_dir)
                data, data_nanformat, _, _ = preprocess_pipeline(
                    data, modality, mod_outputs_dir
                )

                X_nanformat = data_nanformat.reset_index(drop=True)
                X = data.reset_index(drop=True)
                y = label.reset_index(drop=True).values.ravel()
                print("Number of samples/selected features: ", X.shape)

                for fs in self.args.fs_names_list:
                    setup = modality + "_" + fs + "_" + self.args.model_type
                    print("Setup: ", setup)

                    outputs_dir_fs = os.path.join(mod_outputs_dir, fs)
                    outputs_dir_fs_model = os.path.join(
                        mod_outputs_dir, fs + "_" + self.args.model_type
                    )

                    if not os.path.exists(outputs_dir_fs_model):
                        print("Skipping: ", outputs_dir_fs_model)
                        continue

                    # get the selected features of the classifier model
                    feature_mask_df = pd.read_excel(
                        os.path.join(outputs_dir_fs, "Selected_features.xlsx")
                    )
                    assert (
                        feature_mask_df["Feature_names"] == X.columns
                    ).all(), "Feature columns of data and feature mask do not match"
                    feature_mask = (
                        feature_mask_df["Selected_features"].to_numpy().astype(bool)
                    )

                    X_f = X_nanformat.iloc[:, feature_mask]

                    model_predictions_dir = os.path.join(
                        outputs_dir_fs_model, "Model_predictions.xlsx"
                    )
                    try:
                        model_predictions = pd.read_excel(model_predictions_dir)
                    except:
                        print(
                            "Model checkpoint not available yet, skipping ",
                            self.args.model_type,
                        )
                        continue

                    # get the model predictions for all outer folds
                    n_models = model_predictions.shape[0]
                    for i in range(n_models):
                        model_pkl_i = joblib.load(
                            os.path.join(
                                outputs_dir_fs_model,
                                "Pickled_model",
                                "joblib_model_with_info_outer%s.pkl" % i,
                            )
                        )
                        model_i = model_pkl_i["Model"]

                        # get datasplit of the current model (stored in Model_predictions)
                        # remove multi-white spaces, cast "[x1,x2]" to array format
                        split_idx = ast.literal_eval(
                            re.sub("\\s+", " ", model_predictions["test"][i])
                            .replace("[ ", "[")
                            .replace(" ", ",")
                        )
                        split = [
                            "val" if s in split_idx else "train"
                            for s in X_f.index.values
                        ]

                        for s in ["train", "val"]:
                            X_s = X_f.loc[[s_i == s for s_i in split]]
                            y_s = y[[s_i == s for s_i in split]]

                            pred_proba = model_i.predict_proba(X_s)[
                                :, 1
                            ]  # get predictions for train/val samples
                            prob_thresh = self.args.threshold
                            pred = (pred_proba >= prob_thresh).astype(bool)

                            # get performance scores
                            scores_dict = self.get_performance_scores(
                                pred_proba, pred, y_s, confidence_interval=False
                            )
                            cv_eval_df = pd.concat(
                                [
                                    cv_eval_df.reset_index(drop=True),
                                    pd.DataFrame(
                                        dict(
                                            {
                                                "model": setup,
                                                "Modality, Model": modality_txt
                                                + self.args.model_type,
                                                "Feature selection": fs,
                                                "fold": i,
                                                "split": s,
                                            },
                                            **scores_dict
                                        ),
                                        index=[0],
                                    ),
                                ],
                                axis=0,
                            )

                    # report performance scores and confidence intervals (computed as lower/upper quantiles of n outer folds)
                    for s in ["train", "val"]:
                        tmp_df = cv_eval_df.loc[
                            (cv_eval_df["model"] == setup) & (cv_eval_df["split"] == s)
                        ]

                        cv_eval_summary_df = pd.concat(
                            [
                                cv_eval_summary_df.reset_index(drop=True),
                                pd.DataFrame(
                                    {
                                        "model": setup,
                                        "Modality, Model": modality_txt
                                        + self.args.model_type,
                                        "Feature selection": fs,
                                        "split": s,
                                        "f1_score mean": tmp_df["f1_score"].mean(),
                                        "f1_score std": tmp_df["f1_score"].std(),
                                        "f1_score lower": tmp_df["f1_score"].quantile(
                                            self.args.ci[0]
                                        ),
                                        "f1_score upper": tmp_df["f1_score"].quantile(
                                            self.args.ci[1]
                                        ),
                                        "accuracy mean": tmp_df["accuracy"].mean(),
                                        "accuracy std": tmp_df["accuracy"].std(),
                                        "accuracy lower": tmp_df["accuracy"].quantile(
                                            self.args.ci[0]
                                        ),
                                        "accuracy upper": tmp_df["accuracy"].quantile(
                                            self.args.ci[1]
                                        ),
                                        "sensitivity mean": tmp_df[
                                            "sensitivity"
                                        ].mean(),
                                        "sensitivity std": tmp_df["sensitivity"].std(),
                                        "sensitivity lower": tmp_df[
                                            "sensitivity"
                                        ].quantile(self.args.ci[0]),
                                        "sensitivity upper": tmp_df[
                                            "sensitivity"
                                        ].quantile(self.args.ci[1]),
                                        "specificity mean": tmp_df[
                                            "specificity"
                                        ].mean(),
                                        "specificity std": tmp_df["specificity"].std(),
                                        "specificity lower": tmp_df[
                                            "specificity"
                                        ].quantile(self.args.ci[0]),
                                        "specificity upper": tmp_df[
                                            "specificity"
                                        ].quantile(self.args.ci[1]),
                                        "roc_auc mean": tmp_df["roc_auc"].mean(),
                                        "roc_auc std": tmp_df["roc_auc"].std(),
                                        "roc_auc lower": tmp_df["roc_auc"].quantile(
                                            self.args.ci[0]
                                        ),
                                        "roc_auc upper": tmp_df["roc_auc"].quantile(
                                            self.args.ci[1]
                                        ),
                                    },
                                    index=[0],
                                ),
                            ],
                            axis=0,
                        )

            # rank the model setups according to the F1 score metric
            ranking_metric = "f1_score"
            model_ranking = [
                m
                for m in cv_eval_summary_df.loc[cv_eval_summary_df["split"] == "val"]
                .sort_values(by=ranking_metric + " mean", ascending=False)["model"]
                .values
            ]
            print("Model validation score ranking: \n", model_ranking)

            # statistical test comparing all models to the best performing one
            best_model_name = model_ranking[0]
            df2 = cv_eval_df.loc[
                (cv_eval_df["model"] == best_model_name)
                & (cv_eval_df["split"] == "val")
            ]
            for m in model_ranking[1::]:
                df1 = cv_eval_df.loc[
                    (cv_eval_df["model"] == m) & (cv_eval_df["split"] == "val")
                ]

                w, p = stats.wilcoxon(
                    pd.to_numeric(df1[ranking_metric]),
                    pd.to_numeric(df2[ranking_metric]),
                    alternative="two-sided",
                )
                cv_eval_summary_df.loc[
                    (cv_eval_summary_df["model"] == m)
                    & (cv_eval_summary_df["split"] == "val"),
                    "wilcoxon test p-value",
                ] = p
                cv_eval_summary_df.loc[
                    (cv_eval_summary_df["model"] == m)
                    & (cv_eval_summary_df["split"] == "val"),
                    "wilcoxon test stats",
                ] = w

            # cv_eval_df.to_excel(
            #     os.path.join(
            #         summary_outputs_dir,
            #         self.args.filename_prefix
            #         + "cv_evaluation.xlsx",
            #     )
            # )
            cv_eval_summary_df.to_excel(
                os.path.join(
                    summary_outputs_dir,
                    self.args.filename_prefix + "cv_evaluation_summary.xlsx",
                )
            )

        if self.args.evaluate_test:

            print("Start testset evaluation")
            print("Loading file: ", self.args.data_dir)

            eval_df = pd.DataFrame(
                columns=[
                    "model",
                    "Modality, Model",
                    "Feature selection",
                    "split",
                    "f1_score",
                    "accuracy",
                    "sensitivity",
                    "specificity",
                    "roc_auc",
                ]
            )
            eval_df_ci = pd.DataFrame(
                columns=[
                    "model",
                    "Modality, Model",
                    "Feature selection",
                    "split",
                    "f1_score",
                    "accuracy",
                    "sensitivity",
                    "specificity",
                    "roc_auc",
                ]
            )
            eval_dfs_bs = {}

            for modality, modality_txt in zip(
                self.args.modalities, self.args.modality_txts
            ):

                mod_outputs_dir = os.path.join(self.args.outputs_dir, modality)
                summary_outputs_dir = os.path.dirname(mod_outputs_dir)

                # load the data
                data, label = load_data(self.args.data_dir)
                data, data_nanformat, _, _ = preprocess_pipeline(
                    data, modality, mod_outputs_dir
                )

                X_nanformat = data_nanformat.reset_index(drop=True)
                X = data.reset_index(drop=True)
                y = label.reset_index(drop=True).values.ravel()
                print("Number of samples/selected features: ", X.shape)

                for fs in self.args.fs_names_list:
                    setup = modality + "_" + fs + "_" + self.args.model_type
                    print("Setup: ", setup)
                    ensemble_pred_df = pd.DataFrame(
                        columns=[
                            "fold",
                            "setup",
                            "sample",
                            "split",
                            "label",
                            "pred_proba",
                            "pred",
                        ]
                    )

                    outputs_dir_fs = os.path.join(mod_outputs_dir, fs)
                    outputs_dir_fs_model = os.path.join(
                        mod_outputs_dir, fs + "_" + self.args.model_type
                    )

                    if not os.path.exists(outputs_dir_fs_model):
                        print("Skipping: ", outputs_dir_fs_model)
                        continue

                    # get the selected features of the classifier model
                    feature_mask_df = pd.read_excel(
                        os.path.join(outputs_dir_fs, "Selected_features.xlsx")
                    )
                    assert (
                        feature_mask_df["Feature_names"] == X.columns
                    ).all(), "Feature columns of data and feature mask do not match"
                    feature_mask = (
                        feature_mask_df["Selected_features"].to_numpy().astype(bool)
                    )

                    X_f = X_nanformat.iloc[:, feature_mask]

                    model_predictions_dir = os.path.join(
                        outputs_dir_fs_model, "Model_predictions.xlsx"
                    )
                    try:
                        model_predictions = pd.read_excel(model_predictions_dir)
                    except:
                        print(
                            "Model checkpoint not available yet, skipping ",
                            self.args.model_type,
                        )
                        continue

                    # get the model predictions for all outer folds
                    n_models = model_predictions.shape[0]
                    for i in range(n_models):
                        model_pkl_i = joblib.load(
                            os.path.join(
                                outputs_dir_fs_model,
                                "Pickled_model",
                                "joblib_model_with_info_outer%s.pkl" % i,
                            )
                        )
                        model_i = model_pkl_i["Model"]

                        pred_proba = model_i.predict_proba(X_f)[:, 1]
                        prob_thresh = self.args.threshold
                        pred = (pred_proba >= prob_thresh).astype(bool)

                        split = ["test" for j in range(y.shape[0])]

                        ensemble_pred_df = pd.concat(
                            [
                                ensemble_pred_df.reset_index(drop=True),
                                pd.DataFrame(
                                    {
                                        "fold": (np.ones(X_f.shape[0]) * i).astype(
                                            np.uint8
                                        ),
                                        "setup": setup,
                                        "sample": data.index,
                                        "split": split,
                                        "label": y,
                                        "pred_proba": pred_proba,
                                        "pred": pred,
                                    }
                                ),
                            ],
                            axis=0,
                        )

                    # get an ensemble prediction (group dataframe by sample idx, then average predictions)
                    s = "test"
                    group_df = ensemble_pred_df.loc[
                        ensemble_pred_df["split"] == s
                    ].groupby(["sample"])
                    mean_proba_per_sample = group_df["pred_proba"].mean().values
                    std_proba_per_sample = group_df["pred_proba"].std().values
                    vote_pred_per_sample = (
                        group_df["pred_proba"].mean().values >= prob_thresh
                    ).astype(np.uint8)
                    label_per_sample = (
                        group_df["label"].unique().values.astype(np.uint8)
                    )
                    idx = group_df.mean().index.get_level_values(level=0).values

                    # save the model predictions to file
                    ensemble_final_pred_df = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "sample": idx,
                                    "split": [s for k in range(len(idx))],
                                    "pred_proba_mean": mean_proba_per_sample,
                                    "pred_proba_std": std_proba_per_sample,
                                    "pred": vote_pred_per_sample,
                                    "label": label_per_sample,
                                }
                            )
                        ],
                        axis=0,
                    )
                    # ensemble_final_pred_df.to_excel(
                    #     os.path.join(
                    #         outputs_dir_fs_model,
                    #         self.args.filename_prefix
                    #         + "Ensemble_predictions_testset.xlsx",
                    #     )
                    # )

                    # get performance scores with and without bootstrapping (required for confidence interval computation)
                    scores_dict = self.get_performance_scores(
                        mean_proba_per_sample,
                        vote_pred_per_sample,
                        label_per_sample,
                        confidence_interval=False,
                    )
                    scores_dict_ci, df_bs = self.get_performance_scores(
                        mean_proba_per_sample,
                        vote_pred_per_sample,
                        label_per_sample,
                        confidence_interval=True,
                        return_bootstraps=True,
                    )
                    eval_dfs_bs[setup] = df_bs

                    eval_df = pd.concat(
                        [
                            eval_df.reset_index(drop=True),
                            pd.DataFrame(
                                dict(
                                    {
                                        "model": setup,
                                        "Modality, Model": modality_txt
                                        + self.args.model_type,
                                        "Feature selection": fs,
                                        "split": s,
                                    },
                                    **scores_dict
                                ),
                                index=[0],
                            ),
                        ],
                        axis=0,
                    )
                    eval_df.to_excel(
                        os.path.join(
                            summary_outputs_dir,
                            self.args.filename_prefix
                            + "ensemble_evaluation_summary.xlsx",
                        )
                    )

                    eval_df_ci = pd.concat(
                        [
                            eval_df_ci.reset_index(drop=True),
                            pd.DataFrame(
                                dict(
                                    {
                                        "model": setup,
                                        "Modality, Model": modality_txt
                                        + self.args.model_type,
                                        "Feature selection": fs,
                                        "split": s,
                                    },
                                    **scores_dict_ci
                                ),
                                index=[0],
                            ),
                        ],
                        axis=0,
                    )
                    eval_df_ci.to_excel(
                        os.path.join(
                            summary_outputs_dir,
                            self.args.filename_prefix
                            + "ensemble_evaluation_summary_CI.xlsx",
                        )
                    )

                    # visualize the probability distribution
                    probability_histplot(
                        label_per_sample,
                        mean_proba_per_sample,
                        ax=None,
                        fname=os.path.join(
                            outputs_dir_fs_model,
                            self.args.filename_prefix
                            + "ensemble_probabilities_"
                            + s
                            + ".jpg",
                        ),
                    )
                    probability_scatterplot(
                        label_per_sample,
                        mean_proba_per_sample,
                        std_proba_per_sample,
                        ax=None,
                        fname=os.path.join(
                            outputs_dir_fs_model,
                            self.args.filename_prefix
                            + "ensemble_probabilities_std_"
                            + s
                            + ".jpg",
                        ),
                    )

                    # get BTI-RADS scores using the fitted thresholds
                    if self.args.evaluate_btirads:
                        BTIRADS_classifier = BTIRADS2(self.args)
                        thresholds = np.load(
                            os.path.join(outputs_dir_fs_model, "btirads_thresholds.npy")
                        )

                        btirads_pd = BTIRADS_classifier.btirads_classification(
                            mean_proba_per_sample, thresholds
                        )
                        btirads_pd.index = idx
                        btirads_pd["Pred"] = vote_pred_per_sample
                        btirads_pd["Label"] = label_per_sample

                        btirads_pd = btirads_pd.rename(
                            columns={"pred_proba": "Probability"}
                        )

                        # changing column order
                        btirads_pd = btirads_pd[
                            ["Label", "Pred", "BTI-RADS grading", "Probability"]
                        ]
                        btirads_pd.to_excel(
                            os.path.join(
                                outputs_dir_fs_model,
                                self.args.filename_prefix + "BTIRADS_scores.xlsx",
                            )
                        )

                        # append features for inspection
                        for f in data_nanformat.columns.values:
                            btirads_pd[f] = data_nanformat[f].loc[idx]

                        btirads_pd.to_excel(
                            os.path.join(
                                outputs_dir_fs_model,
                                self.args.filename_prefix
                                + "BTIRADS_scores_and_features.xlsx",
                            )
                        )

                        # get malignancy frequencies without 95% CI
                        malignancy_frequency, malignancy_sensitivity = (
                            BTIRADS_classifier.get_malignancy_frequencies(
                                btirads_pd, label_per_sample, confidence_interval=False
                            )
                        )
                        malignancy_frequency.to_excel(
                            os.path.join(
                                outputs_dir_fs_model,
                                self.args.filename_prefix
                                + "BTIRADS_malignancy_frequency.xlsx",
                            )
                        )

                        # get malignancy frequencies with 95% CI
                        malignancy_frequency, malignancy_sensitivity = (
                            BTIRADS_classifier.get_malignancy_frequencies(
                                btirads_pd, label_per_sample, confidence_interval=True
                            )
                        )
                        malignancy_frequency.to_excel(
                            os.path.join(
                                outputs_dir_fs_model,
                                self.args.filename_prefix
                                + "BTIRADS_malignancy_frequency_CI.xlsx",
                            )
                        )

            # Wilcoxon rank test to compare models to best model
            ref_model = self.args.ref_model
            for k in [k2 for k2 in eval_dfs_bs.keys() if k2 != ref_model]:
                w, p = stats.wilcoxon(
                    pd.to_numeric(eval_dfs_bs[k]["f1_score"]),
                    pd.to_numeric(eval_dfs_bs[ref_model]["f1_score"]),
                    alternative="two-sided",
                )

                eval_df_ci.loc[
                    eval_df_ci["model"] == k, "Wilcoxon p-value (best model)"
                ] = p
                eval_df_ci.loc[
                    eval_df_ci["model"] == k, "Wilcoxon stats (best model)"
                ] = w

            eval_df_ci.to_excel(
                os.path.join(
                    summary_outputs_dir,
                    self.args.filename_prefix + "ensemble_evaluation_summary_CI.xlsx",
                )
            )
