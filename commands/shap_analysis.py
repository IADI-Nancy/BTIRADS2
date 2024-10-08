import os
import pandas as pd
from utils.data_io import load_data
from utils.preprocessing import preprocess_pipeline
import joblib
import numpy as np
from utils.format_utils import feature_names_printed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", message=".*fitted without feature names*")
warnings.filterwarnings(
    "ignore",
    message=".*Dropping invalid columns in DataFrameGroupBy.mean is deprecated*",
)
warnings.filterwarnings("ignore", message=".*ntree_limit is deprecated*")
warnings.filterwarnings(
    "ignore", message=".*The 'nopython' keyword.*"
)  # https://github.com/slundberg/shap/issues/2909
import shap


class Shap_analysis:
    def __init__(self, args) -> None:
        self.args = args
        self.probability_space = True

    def exec(self):
        print("Start shap analysis")
        print("Loading file: ", self.args.data_dir)

        modality = self.args.modality
        fs = self.args.fs_name
        model_type = self.args.model_type
        setup = modality + "_" + fs + "_" + model_type
        print("Setup: ", setup)

        mod_outputs_dir = os.path.join(self.args.outputs_dir, modality)
        outputs_dir_fs = os.path.join(mod_outputs_dir, fs)
        outputs_dir_fs_model = os.path.join(mod_outputs_dir, fs + "_" + model_type)

        # load the data
        data_orig, label = load_data(self.args.data_dir)
        data, data_nanformat, features_numeric, _ = preprocess_pipeline(
            data_orig.copy(), modality, mod_outputs_dir
        )

        X_nanformat = data_nanformat.reset_index(drop=True)
        X = data.reset_index(drop=True)
        y = label.reset_index(drop=True).values.ravel()

        # get the selected features of the desired model setup
        feature_mask_df = pd.read_excel(
            os.path.join(outputs_dir_fs, "Selected_features.xlsx")
        )
        assert (
            feature_mask_df["Feature_names"] == X.columns
        ).all(), "Feature columns of data and feature mask do not match"
        feature_mask = feature_mask_df["Selected_features"].to_numpy().astype(bool)

        X_f = X_nanformat.iloc[:, feature_mask]
        X_f_plot = X_f.copy()
        X_f_plot[[f for f in features_numeric if f in X_f_plot.columns]] = data_orig[
            [f for f in features_numeric if f in X_f_plot.columns]
        ].values

        # Get feature names in readable format
        feature_names = list(X_f.rename(columns=feature_names_printed).columns.values)
        n_features_plot = len(feature_names) + 1
        feature_display_range = slice(-1, -n_features_plot, -1)

        ensemble_pred_df = pd.DataFrame(
            columns=["checkpoint", "fold", "sample", "label", "pred_proba"]
        )

        shap_values_list = []
        expected_value_list = []
        checkpoint = 0

        model_predictions_dir = os.path.join(
            outputs_dir_fs_model, "Model_predictions.xlsx"
        )
        try:
            model_predictions = pd.read_excel(model_predictions_dir)
        except:
            raise "Model checkpoint not available"

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
            clf_i = model_i["classifier"]

            X_f_preproc = model_i[:-1].transform(X_f)
            pred_proba = model_i.predict_proba(X_f)[:, 1]
            ensemble_pred_df = pd.concat(
                [
                    ensemble_pred_df.reset_index(drop=True),
                    pd.DataFrame(
                        {
                            "checkpoint": checkpoint,
                            "fold": (np.ones(X_f.shape[0]) * i).astype(np.uint8),
                            "sample": data.index,
                            "label": y,
                            "pred_proba": pred_proba,
                        }
                    ),
                ],
                axis=0,
            )

            if self.probability_space:
                explainer = shap.TreeExplainer(
                    clf_i,
                    feature_names=feature_names,
                    model_output="probability",
                    data=X_f_preproc,
                )
            else:
                explainer = shap.TreeExplainer(clf_i, feature_names=feature_names)

            shap_values = explainer.shap_values(X_f_preproc)
            expected_value = explainer.expected_value
            shap_values_list.append(shap_values)
            expected_value_list.append(expected_value)
            checkpoint += 1

        global_persample_shap_values = np.stack(
            shap_values_list
        )  # (n_checkpoints, n_samples, n_features)
        mean_persample_shap = global_persample_shap_values.mean(
            0
        )  # mean over models/runs -> (n_samples, n_features)

        global_expected_value = np.stack(expected_value_list)  # (n_checkpoints, 1)
        mean_expected_value = global_expected_value.mean(
            0
        )  # mean over models/runs per feature

        f = plt.figure()
        summary_plot = shap.summary_plot(
            mean_persample_shap,
            X_f,
            feature_names=feature_names,
            show=False,
            max_display=n_features_plot,
            plot_size=(13, 10),
        )
        ax = plt.gca()
        f.savefig(
            os.path.join(
                outputs_dir_fs_model, self.args.filename_prefix + "shap_beeswarm.png"
            ),
            bbox_inches="tight",
            dpi=1200,
        )
        plt.close(f)

        feature_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean absolute shap value": np.mean(
                    np.abs(mean_persample_shap), axis=0
                ),
            }
        )
        feature_importance = feature_importance.sort_values(
            "Mean absolute shap value", ascending=False
        )

        plt.figure(figsize=(11, 10))
        color_blue = shap.plots.colors.blue_rgb
        ax = sns.barplot(
            data=feature_importance,
            x="Mean absolute shap value",
            y="Feature",
            color=color_blue,
        )
        ax.bar_label(ax.containers[0], fmt="%.4f", fontsize=8, padding=2, color="gray")
        plt.gca().xaxis.set_ticks_position("bottom")
        plt.gca().yaxis.set_ticks_position("none")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().tick_params("x", labelsize=9)
        plt.gca().tick_params("y", labelsize=9)
        plt.gca().xaxis.label.set_size(10)
        plt.gca().yaxis.label.set_size(10)

        plt.savefig(
            os.path.join(
                outputs_dir_fs_model,
                self.args.filename_prefix + "shap_feature_importance_barplot.png",
            ),
            bbox_inches="tight",
            dpi=1200,
        )  # , dpi=600)
        plt.close(f)

        group_df = ensemble_pred_df
        group_df = group_df.groupby(["sample"], sort=False)
        mean_proba_per_sample = group_df.mean()["pred_proba"]
        std_proba_per_sample = group_df.std()["pred_proba"]
        label_per_sample = group_df["label"].unique().astype(np.uint8)

        # decision plot per sample of interest
        for sample_idx in self.args.sample_list:
            save_txt = (
                str(sample_idx)
                + "_target"
                + str(np.round(label_per_sample.loc[sample_idx], 2))
                + "_predproba"
                + str(np.round(mean_proba_per_sample.loc[sample_idx], 2))
                + "_std"
                + str(np.round(std_proba_per_sample.loc[sample_idx], 2))
            )
            sample_i = np.nonzero(data.index == sample_idx)[0][0]

            shap_object = shap.Explanation(
                values=np.round(mean_persample_shap[sample_i, :], 2),
                base_values=np.round(mean_expected_value, 2),
                data=np.round(X_f.iloc[sample_i, :].values, 2),
                feature_names=feature_names,
            )

            f = plt.figure()
            if self.probability_space:
                decision_plot = shap.decision_plot(
                    np.round(mean_expected_value, 2),
                    np.round(mean_persample_shap[sample_i, :], 2),
                    np.round(X_f_plot.iloc[sample_i, :].values, 2),
                    feature_names=feature_names,
                    show=False,
                    xlim=[0, 1],
                    feature_display_range=feature_display_range,  # feature_order=feature_order,
                )
            else:
                decision_plot = shap.decision_plot(
                    np.round(mean_expected_value, 2),
                    np.round(mean_persample_shap[sample_i, :], 2),
                    np.round(X_f_plot.iloc[sample_i, :].values, 2),
                    feature_names=feature_names,
                    show=False,
                    link="logit",
                    feature_display_range=feature_display_range,  # feature_order=feature_order,
                )

            plt.savefig(
                os.path.join(
                    outputs_dir_fs_model,
                    self.args.filename_prefix
                    + "shap_decisionplot_"
                    + save_txt
                    + ".png",
                ),
                bbox_inches="tight",
                dpi=1200,
            )
            plt.close(f)
