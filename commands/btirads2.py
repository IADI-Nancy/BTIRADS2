import os
import pandas as pd
from utils.data_io import load_data
from utils.preprocessing import preprocess_pipeline
from utils.visualization import probability_histplot, probability_scatterplot
import joblib
import numpy as np
import ast
import re
from sklearn.neighbors import KernelDensity
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore", message=".*fitted without feature names*")
warnings.filterwarnings(
    "ignore",
    message=".*Dropping invalid columns in DataFrameGroupBy.mean is deprecated*",
)


class BTIRADS2:
    def __init__(self, args) -> None:
        self.args = args

    def btirads_classification(self, pred_proba, thresholds):

        btirads_pd = pd.DataFrame(
            {"pred_proba": pred_proba, "BTI-RADS grading": np.zeros(pred_proba.shape)}
        )
        thresholds = np.concatenate([np.zeros(1), thresholds])

        for c in range(len(thresholds) - 1):
            if thresholds[c] == 0:
                pos_samples = (pred_proba >= thresholds[c]) & (
                    pred_proba <= thresholds[c + 1]
                )
            else:
                pos_samples = (pred_proba > thresholds[c]) & (
                    pred_proba <= thresholds[c + 1]
                )
            btirads_pd.loc[pos_samples, "BTI-RADS grading"] = int(
                c + 2
            )  # start counting at 2

        return btirads_pd

    def get_malignancy_scores(self, btirads_df, label):

        data_label_joint = pd.concat(
            [btirads_df.reset_index(), label.reset_index()], axis=1
        )

        scores = pd.DataFrame(
            columns=[
                "BTI-RADS grading",
                "Total n",
                "Benign tumors n",
                "Malignant tumors n",
                "Benign tumors %",
                "Malignant tumors %",
            ],
            dtype="float",
        )

        for f in btirads_df.columns:
            feature_count = data_label_joint[f].sum()
            benign_count = data_label_joint.loc[data_label_joint["Label"] == 0][f].sum()
            malignant_count = data_label_joint.loc[data_label_joint["Label"] == 1][
                f
            ].sum()
            if feature_count > 0:
                benign_perc = np.round(100 * (benign_count / feature_count), 2)
                malignant_perc = np.round(100 * (malignant_count / feature_count), 2)
            else:
                benign_perc = 0
                malignant_perc = 0

            scores = pd.concat(
                [
                    scores,
                    pd.DataFrame(
                        {
                            "BTI-RADS grading": f,
                            "Total n": feature_count,
                            "Benign tumors n": benign_count,
                            "Malignant tumors n": malignant_count,
                            "Benign tumors %": benign_perc,
                            "Malignant tumors %": malignant_perc,
                        },
                        index=[0],
                    ),
                ],
                axis=0,
                ignore_index=True,
            )

        return scores

    def get_malignancy_frequencies(self, btirads_df, label, confidence_interval=True):

        if not isinstance(label, pd.DataFrame):
            label = pd.DataFrame({"Label": label})

        enc = OneHotEncoder(
            categories="auto", dtype=np.float64, handle_unknown="ignore", sparse=False
        ).fit(pd.DataFrame(btirads_df["BTI-RADS grading"], dtype="int"))
        btirads_df = pd.DataFrame(
            columns=enc.get_feature_names_out(),
            data=enc.transform(pd.DataFrame(btirads_df["BTI-RADS grading"])),
        )

        if not confidence_interval:
            scores = self.get_malignancy_scores(btirads_df, label)

            malignancy_sensitivity = (
                scores.loc[
                    scores["BTI-RADS grading"] == "BTI-RADS grading_4",
                    "Malignant tumors n",
                ].iloc[0]
                + scores.loc[
                    scores["BTI-RADS grading"] == "BTI-RADS grading_5",
                    "Malignant tumors n",
                ]
            ).iloc[0] / scores["Malignant tumors n"].sum()

        # confidence intervals using test set bootstrapping
        else:
            rng = np.random.RandomState(seed=self.args.random_seed)
            bidx = np.arange(label.shape[0])

            bootstrap_df = pd.DataFrame()
            malignancy_sensitivities = []
            for b in range(self.args.n_bootstrap):
                pred_bidx = rng.choice(bidx, size=bidx.shape[0], replace=True)
                scores_df = self.get_malignancy_scores(
                    btirads_df.iloc[pred_bidx], label.iloc[pred_bidx]
                )
                malignancy_sensitivity = (
                    scores_df.loc[
                        scores_df["BTI-RADS grading"] == "BTI-RADS grading_4",
                        "Malignant tumors n",
                    ].iloc[0]
                    + scores_df.loc[
                        scores_df["BTI-RADS grading"] == "BTI-RADS grading_5",
                        "Malignant tumors n",
                    ]
                ).iloc[0] / scores_df["Malignant tumors n"].sum()
                malignancy_sensitivities.append(malignancy_sensitivity)
                bootstrap_df = pd.concat([bootstrap_df, scores_df])

            bootstrap_mean_scores = (
                bootstrap_df.groupby("BTI-RADS grading").mean().round().astype(int)
            )
            bootstrap_lower_scores = (
                bootstrap_df.groupby("BTI-RADS grading")
                .quantile(self.args.ci[0])
                .round()
                .astype(int)
            )
            bootstrap_upper_scores = (
                bootstrap_df.groupby("BTI-RADS grading")
                .quantile(self.args.ci[1])
                .round()
                .astype(int)
            )

            scores = bootstrap_mean_scores
            for g in scores.index:
                for k in scores.columns.values:
                    scores.loc[g, k] = (
                        str(bootstrap_mean_scores.loc[g, k])
                        + " ["
                        + str(bootstrap_lower_scores.loc[g, k])
                        + ","
                        + str(bootstrap_upper_scores.loc[g, k])
                        + "]"
                    )

            scores = scores.reset_index()
            malignancy_sensitivities = np.array(malignancy_sensitivities)
            malignancy_sensitivity = (
                str(np.round(np.mean(malignancy_sensitivities) * 100))
                + " ["
                + str(np.round(np.quantile(malignancy_sensitivities, 0.025) * 100))
                + ","
                + str(np.round(np.quantile(malignancy_sensitivities, 0.975) * 100))
                + "]"
            )

        malignancy_sensitivity = pd.DataFrame(
            {"malignancy sensitivity": malignancy_sensitivity}, index=[0]
        )

        return scores, malignancy_sensitivity

    def integ_func(self, kde, x1):

        def f_kde(x):
            return np.exp((kde.score_samples([[x]])))

        res, err = integrate.quad(f_kde, -np.inf, x1)

        return res

    def kernel_density_threshold(
        self,
        label,
        pred_proba,
        ax=None,
        fname=None,
        bandwidth=0.05,
        thresholds=[0.001, 0.05, 0.25],
    ):

        # fit the malignant class kernel density function
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            pred_proba[label == 1].reshape(-1, 1)
        )

        # sample the integral at N positions
        X = np.linspace(0, 1, 1000).reshape(-1, 1)
        sampled_log_dens = kde.score_samples(X)  # score_sample returns log pdf -> exp
        samples_dens = np.exp(sampled_log_dens)

        plt.figure()
        plt.plot(X, samples_dens, "black")
        plt.savefig(fname, dpi=1200)
        plt.close()

        sampled_integrals = np.array([self.integ_func(kde, x1) for x1 in X])

        # get x where user selected malignancy thresholds are satisfied
        thresh_mc = []
        for t in thresholds:
            x_t = np.argmin(np.abs(sampled_integrals - t))
            thresh_mc.append(np.round(X[x_t], 2).item())

        thresh_mc += [1.0]

        plt.figure()
        plt.plot(X, sampled_integrals, "black")
        plt.axvline(
            x=thresh_mc[0], color="black", ls="--", lw=2
        )  # , label='BTI-RADS 3'
        plt.axvline(x=thresh_mc[1], color="black", ls="--", lw=2)
        plt.axvline(x=thresh_mc[2], color="black", ls="--", lw=2)
        plt.xlabel("Model output (probability)")
        plt.ylabel("Kernel density integral")
        plt.savefig(fname.replace(".jpg", "_integral.jpg"), dpi=1200)
        plt.close()

        return thresh_mc

    def exec(self):
        print("Start fitting BTI-RADS thresholds")
        print("Loading file: ", self.args.data_dir)

        modality = self.args.modality
        mod_outputs_dir = os.path.join(self.args.outputs_dir, modality)

        # load the data
        data, label = load_data(self.args.data_dir)

        # preprocess the data
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
            feature_mask = feature_mask_df["Selected_features"].to_numpy().astype(bool)

            X_f = X_nanformat.iloc[:, feature_mask]

            # get the model predictions
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

            # get per model predictions for train/val samples of each fold
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
                split = ["val" if s in split_idx else "train" for s in X_f.index.values]

                pred = model_i.predict(X_f)
                pred_proba = model_i.predict_proba(X_f)
                ensemble_pred_df = pd.concat(
                    [
                        ensemble_pred_df.reset_index(drop=True),
                        pd.DataFrame(
                            {
                                "fold": (np.ones(X_f.shape[0]) * i).astype(np.uint8),
                                "setup": setup,
                                "sample": data.index,
                                "split": split,
                                "label": y,
                                "pred_proba": pred_proba[:, 1],
                                "pred": pred,
                            }
                        ),
                    ],
                    axis=0,
                )

            # for the validation set samples, get an ensemble prediction using averaging
            s = "val"
            group_df = ensemble_pred_df.loc[ensemble_pred_df["split"] == s].groupby(
                ["sample"]
            )
            mean_proba_per_sample = group_df["pred_proba"].mean().values
            std_proba_per_sample = group_df["pred_proba"].std().values
            vote_pred_per_sample = (
                group_df["pred"].mean().values >= self.args.threshold
            ).astype(np.uint8)
            label_per_sample = group_df["label"].unique().values.astype(np.uint8)
            idx = group_df.mean().index.get_level_values(level=0).values

            # visualize the probability distributions
            probability_histplot(
                label_per_sample,
                mean_proba_per_sample,
                ax=None,
                fname=os.path.join(
                    outputs_dir_fs_model, "ensemble_probabilities_" + s + ".jpg"
                ),
            )
            probability_scatterplot(
                label_per_sample,
                mean_proba_per_sample,
                std_proba_per_sample,
                ax=None,
                fname=os.path.join(
                    outputs_dir_fs_model, "ensemble_probabilities_std_" + s + ".jpg"
                ),
            )

            # fit the btirads2 thresholds on the validation set distribution and save to file
            thresholds = self.kernel_density_threshold(
                label_per_sample,
                mean_proba_per_sample,
                ax=None,
                fname=os.path.join(
                    outputs_dir_fs_model, "btirads_kernel_density_" + s + ".jpg"
                ),
            )
            np.save(
                os.path.join(outputs_dir_fs_model, "btirads_thresholds.npy"), thresholds
            )
            pd.DataFrame(thresholds).to_excel(
                os.path.join(outputs_dir_fs_model, "btirads_thresholds.xlsx"),
                index=False,
            )

            # perform the btirads scoring for the identified thresholds
            btirads_pd = self.btirads_classification(mean_proba_per_sample, thresholds)
            btirads_pd.index = idx
            btirads_pd["Pred"] = vote_pred_per_sample
            btirads_pd["Label"] = label_per_sample
            btirads_pd.to_excel(
                os.path.join(outputs_dir_fs_model, "BTIRADS_scores_" + s + ".xlsx")
            )

            # report malignancy frequencies for the validation set
            malignancy_frequency, malignancy_sensitivity = (
                self.get_malignancy_frequencies(btirads_pd, label_per_sample)
            )
            malignancy_frequency.to_excel(
                os.path.join(
                    outputs_dir_fs_model, "BTIRADS_malignancy_frequency_" + s + ".xlsx"
                )
            )
            malignancy_sensitivity.to_excel(
                os.path.join(
                    outputs_dir_fs_model,
                    "BTIRADS_malignancy_sensitivitys_" + s + ".xlsx",
                )
            )
