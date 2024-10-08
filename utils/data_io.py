import pandas as pd
import numpy as np
import os
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
    log_loss,
)
import imblearn


def load_data(fname, index_col=0):

    data = pd.read_excel(fname, index_col=index_col)

    if "Label" in data.columns.values:
        label = data["Label"]
        data = data.drop("Label", axis=1)
    else:
        label = None

    return data, label


def save_results_fs_cv(save_dir, fs_pipeline, X, y):
    X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    results = fs_pipeline.results
    model_pickle_dir = os.path.join(save_dir, "Pickled_model")
    os.makedirs(model_pickle_dir, exist_ok=True)

    for i, model in enumerate(fs_pipeline.results["model"]):
        train_index = fs_pipeline.results["train"][i]
        test_index = fs_pipeline.results["test"][i]

        fit_dict = {
            "Model": model,
            "X_train": X[train_index],
            "y_train": y[train_index],
            "X_test": X[test_index],
            "y_test": y[test_index],
        }
        dump(
            fit_dict,
            os.path.join(model_pickle_dir, "joblib_model_with_info_outer%s.pkl" % i),
        )

    fit_dict = {
        "Model": model,
        "X_train": X[train_index],
        "y_train": y[train_index],
        "X_test": X[test_index],
        "y_test": y[test_index],
    }
    dump(
        fit_dict,
        os.path.join(model_pickle_dir, "joblib_model_with_info_outer%s.pkl" % i),
    )

    results = {"outer": fs_pipeline.results}


def save_results_cv(save_dir, clf, X, y):
    feature_names = X.columns
    X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    outer_results = clf.outer_results

    outer_results.update(
        {
            "outer_test_f1": [],
            "outer_test_auc": [],
            "outer_test_neg_brier": [],
            "outer_test_neg_ce": [],
            "outer_test_accuracy": [],
            "outer_test_recall": [],
            "outer_test_precision": [],
            "outer_test_sensitivity": [],
            "outer_test_specificity": [],
        }
    )

    model_pickle_dir = os.path.join(save_dir, "Pickled_model")
    os.makedirs(model_pickle_dir, exist_ok=True)

    for i, model in enumerate(clf.outer_pred["model"]):
        train_index = clf.outer_pred["train"][i]
        test_index = clf.outer_pred["test"][i]

        y_true, y_pred = y[test_index], model.predict(X[test_index])

        if len(np.unique(y)) == 2:
            if hasattr(model[-1], "decision_fuction"):
                y_score = model.decision_function(X[clf.outer_pred["test"][i]])
            else:
                y_score = model.predict_proba(X[clf.outer_pred["test"][i]])[
                    :, 1
                ].ravel()

            f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
            auc = roc_auc_score(y_true, y_score, labels=[0, 1])
            recall = recall_score(y_true, y_pred, average="binary", pos_label=1)
            precision = precision_score(y_true, y_pred, average="binary", pos_label=1)
            sensitivity = imblearn.metrics.sensitivity_score(
                y_true, y_pred, average="binary", pos_label=1
            )
            specificity = imblearn.metrics.specificity_score(
                y_true, y_pred, average="binary", pos_label=1
            )
        else:
            y_score = model.predict_proba(X[clf.outer_pred["test"][i]])

            f1 = f1_score(y_true, y_pred, average="macro")
            auc = roc_auc_score(
                y_true, y_score, average="macro", multi_class="ovo", labels=[1, 2, 3]
            )
            recall = recall_score(y_true, y_pred, average="macro")
            precision = precision_score(y_true, y_pred, average="macro")
            sensitivity = imblearn.metrics.sensitivity_score(
                y_true, y_pred, average="macro"
            )
            specificity = imblearn.metrics.specificity_score(
                y_true, y_pred, average="macro"
            )

        neg_brier = -brier_score_loss(y_true, y_score)
        neg_ce = -log_loss(y_true, y_score)
        outer_results["outer_test_f1"].append(f1)
        outer_results["outer_test_auc"].append(auc)
        outer_results["outer_test_neg_brier"].append(neg_brier)
        outer_results["outer_test_neg_ce"].append(neg_ce)
        outer_results["outer_test_accuracy"].append(accuracy_score(y_true, y_pred))
        outer_results["outer_test_recall"].append(recall)
        outer_results["outer_test_precision"].append(precision)
        outer_results["outer_test_sensitivity"].append(sensitivity)
        outer_results["outer_test_specificity"].append(specificity)

        fit_dic = {
            "Model": model,
            "X_train": X[train_index],
            "y_train": y[train_index],
            "X_test": X[test_index],
            "y_test": y[test_index],
            "score": clf.outer_results["outer_test_score"][i],
        }
        dump(
            fit_dic,
            os.path.join(model_pickle_dir, "joblib_model_with_info_outer%s.pkl" % i),
        )

    inner_results_reformated = {
        "inner_Fold": [],
        "params": [],
        "mean_test_score": [],
        "std_test_score": [],
        "mean_train_score": [],
        "std_train_score": [],
    }

    for i, fold_results in enumerate(clf.inner_results):
        for j in range(len(fold_results["params"])):
            inner_results_reformated["inner_Fold"].append(i)
            for key in fold_results.keys():
                inner_results_reformated[key].append(fold_results[key][j])

    results = {
        "outer": clf.outer_results,
        "inner": inner_results_reformated,
        "outer summary": {
            "mean": np.mean(clf.outer_results["outer_test_score"]),
            "std": np.std(clf.outer_results["outer_test_score"]),
        },
    }

    with pd.ExcelWriter(os.path.join(save_dir, "NestedCV_results.xlsx")) as writer:
        for loop in results:
            try:
                df = pd.DataFrame(results[loop])
            except:
                df = pd.DataFrame.from_dict(results[loop], orient="index").transpose()

            df.to_excel(writer, sheet_name=loop)

    # Extract predictions from model
    df = pd.DataFrame(
        {key: clf.outer_pred[key] for key in clf.outer_pred if key != "model"},
        dtype="object",
    )
    df.to_excel(os.path.join(save_dir, "Model_predictions.xlsx"))


def save_results_refit(save_dir, clf, X, y):
    feature_names = X.columns
    X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    outer_results = clf.refit_outer_results

    model_pickle_dir = os.path.join(save_dir, "Pickled_model")
    os.makedirs(model_pickle_dir, exist_ok=True)

    model_refit = clf.best_estimator_

    fit_dic = {"Model": model_refit, "X_train": X, "y_train": y}
    dump(fit_dic, os.path.join(model_pickle_dir, "joblib_model_with_info_refit.pkl"))

    inner_results_reformated = {
        "inner_Fold": [],
        "params": [],
        "mean_test_score": [],
        "std_test_score": [],
        "mean_train_score": [],
        "std_train_score": [],
    }

    for i, fold_results in enumerate(clf.refit_inner_results):
        for j in range(len(fold_results["params"])):
            inner_results_reformated["inner_Fold"].append(i)
            for key in fold_results.keys():
                inner_results_reformated[key].append(fold_results[key][j])

    results = {
        "outer": clf.refit_outer_results,
        "inner": inner_results_reformated,
    }  #'outer summary': {'mean': np.nan, 'std': np.nan} no outer summary as no subset

    with pd.ExcelWriter(
        os.path.join(save_dir, "NestedCV_refit_results.xlsx")
    ) as writer:
        for loop in results:
            try:
                df = pd.DataFrame(results[loop])
            except:
                df = pd.DataFrame.from_dict(results[loop], orient="index").transpose()

            df.to_excel(writer, sheet_name=loop)

    # Extract predictions from model
    df = pd.DataFrame(
        {
            key: clf.refit_outer_pred[key]
            for key in clf.refit_outer_pred
            if (key != "model") and (len(clf.refit_outer_pred[key]) > 0)
        },
        dtype="object",
    )
    df.to_excel(os.path.join(save_dir, "Model_predictions_refit.xlsx"))
