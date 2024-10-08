import os
import pandas as pd
import time
from utils.data_io import (
    load_data,
    save_results_fs_cv,
    save_results_cv,
    save_results_refit,
)
from utils.preprocessing import preprocess_pipeline
import numpy as np
from model.feature_selection import (
    get_fs_dict,
    perform_feature_selection,
    get_feature_mask_from_cv_results,
)
from model.classifier import get_gridsearch_dict, statistical_pipeline
from sklearn.preprocessing import MinMaxScaler


class Train:
    def __init__(self, args) -> None:
        self.args = args

    def exec(self):
        print("Starting the training process")
        print("Loading file: ", self.args.data_dir)

        # define the output directory
        modality = self.args.modality
        mod_outputs_dir = os.path.join(self.args.outputs_dir, modality)
        print("Saving to: ", mod_outputs_dir)

        if not os.path.exists(mod_outputs_dir):
            os.makedirs(mod_outputs_dir)

        # load the data
        data, label = load_data(self.args.data_dir)

        # pre-process the data, returning one-hot encoding with zero missing value encoding
        # for the feature selection pipeline and nan encoding for XGBoost training
        data, data_nanformat, _, _ = preprocess_pipeline(
            data, modality, mod_outputs_dir, is_fitted=False
        )

        # prepare the data arrays for the sklearn pipeline (ignore pandas indices)
        X = data.reset_index(drop=True)
        X_nanformat = data_nanformat.reset_index(drop=True)
        y = label.reset_index(drop=True).values.ravel()
        print("Number of samples/selected features: ", X.shape)

        metric = "f1_score"
        fs_metric = "chi2"
        if self.args.fs_aggregation:
            fs_aggregation = "importance_score"
        else:
            fs_aggregation = None
        model_type = "XGB"
        n_splits_outer = 5
        n_splits_inner = 3
        n_repeats_outer = 10
        n_repeats_inner = 10
        use_scale = True
        save_intermediate = True

        # if feature selection pipeline only, include all features
        # this can be useful if you with to count p-values<0.05
        if self.args.fs_only:
            fs_nfeat_list = [X.shape[1]]

        else:
            fs_nfeat_list = self.args.fs_nfeat_list

        # get feature selection setup dictionary
        fs_dict = get_fs_dict(
            fs_metric,
            fs_nfeat_list,
            fs_aggregation,
            scaler=MinMaxScaler,
            use_scale=use_scale,
            random_state=self.args.random_seed,
            save_intermediate=save_intermediate,
        )

        # get models grid search setup dictionary
        grid_search_dict = get_gridsearch_dict(
            model_type,
            scaler=MinMaxScaler,
            use_scale=use_scale,
            random_state=self.args.random_seed,
        )

        # Perform nested cross-validation
        start_time = time.time()

        model_results = {}
        model_performs = {
            "model_name": [],
            "fs_info": [],
            "mean_test_score": [],
            "std_test_score": [],
            "mean_train_score": [],
            "std_train_score": [],
        }

        # perform feature selection
        for f, fs_name in enumerate(fs_dict):
            n_selected_features = fs_dict[fs_name]["fs_info"]["fs_nfeat"]
            if n_selected_features > X.shape[1]:
                continue
            print("\t\t\t================== %s ================" % (fs_name))
            outputs_dir_fs = os.path.join(mod_outputs_dir, fs_name)

            if not os.path.exists(outputs_dir_fs):
                os.makedirs(outputs_dir_fs)

            print("Starting feature selection")

            # Use X instead if X_nanformat as missing data is not supported for chi2
            cv_results = perform_feature_selection(
                X,
                y,
                fs_dict[fs_name]["pipeline_dict"],
                fs_dict[fs_name]["params_dict"],
                fs_dict[fs_name]["pipeline_options"],
                n_jobs=self.args.n_jobs,
                verbose=self.args.verbose,
            )

            save_results_fs_cv(outputs_dir_fs, cv_results, X, y)

            feature_mask = get_feature_mask_from_cv_results(
                X, n_selected_features, outputs_dir_fs, verbose=self.args.verbose
            )

            df = pd.DataFrame(
                {"Feature_names": X.columns.values, "Selected_features": feature_mask}
            )
            df.to_excel(os.path.join(outputs_dir_fs, "Selected_features.xlsx"))

            # extract the selected features from the df with nan encoding for missing values
            X_f = X_nanformat.iloc[:, feature_mask]

            if not self.args.fs_only:
                print("Starting nested cross-validation")
                for model_name in grid_search_dict:
                    print("\t\t\t================== %s ================" % (model_name))
                    outputs_dir_model = os.path.join(
                        mod_outputs_dir, fs_name + "_" + model_name
                    )

                    if not os.path.exists(outputs_dir_model):
                        os.makedirs(outputs_dir_model)

                    cv_clf = statistical_pipeline(
                        X_f,
                        y,
                        grid_search_dict[model_name]["pipeline_dict"],
                        grid_search_dict[model_name]["params_dict"],
                        grid_search_dict[model_name]["pipeline_options"],
                        random_state=self.args.random_seed,
                        n_jobs=self.args.n_jobs,
                        n_splits_outer=n_splits_outer,
                        n_splits_inner=n_splits_inner,
                        n_repeats_outer=n_repeats_outer,
                        n_repeats_inner=n_repeats_inner,
                        metric=metric,
                        refit_outer=True,
                        verbose=self.args.verbose,
                    )

                    save_results_cv(outputs_dir_model, cv_clf, X_f, y)
                    save_results_refit(outputs_dir_model, cv_clf, X_f, y)

                    model_results[model_name] = cv_clf.outer_results
                    model_performs["model_name"].append(
                        grid_search_dict[model_name]["model_info"]["model_name"]
                    )
                    model_performs["fs_info"].append(
                        fs_dict[fs_name]["fs_info"]["fs_metric"]
                        + "_"
                        + str(fs_dict[fs_name]["fs_info"]["fs_nfeat"]).zfill(3)
                    )
                    model_performs["mean_test_score"].append(
                        np.mean(cv_clf.outer_results["outer_test_score"])
                    )
                    model_performs["std_test_score"].append(
                        np.std(cv_clf.outer_results["outer_test_score"])
                    )
                    model_performs["mean_train_score"].append(
                        np.mean(cv_clf.outer_results["outer_train_score"])
                    )
                    model_performs["std_train_score"].append(
                        np.std(cv_clf.outer_results["outer_train_score"])
                    )

                    model_performs_df = pd.DataFrame(model_performs)
                    model_performs_df.to_excel(
                        os.path.join(mod_outputs_dir, "Model_comparison.xlsx")
                    )

            print(
                "Total Time Nested Cross-validation: %.2f" % (time.time() - start_time)
            )
