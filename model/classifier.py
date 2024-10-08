import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from model.nested_cv import GridSearchNestedCV


def get_gridsearch_dict(
    model_type, scaler=MinMaxScaler, use_scale=True, random_state=None
):
    gridsearch_dict = {}

    if model_type == "LR_L1":
        algo = LogisticRegression
        classifier_options = {
            "penalty": "l1",
            "random_state": random_state,
            "solver": "saga",
            "max_iter": 1e6,
            "class_weight": "balanced",
        }
        classifier_params = {
            "C": np.linspace(0.01, 10, 100)
        }  # np.linspace(1e-3,1e3,10)

    if model_type == "LR_L2":
        algo = LogisticRegression
        classifier_options = {
            "penalty": "l2",
            "random_state": random_state,
            "solver": "saga",
            "max_iter": 1e6,
            "class_weight": "balanced",
        }  # l2 better than l1 when using initial feature selection
        classifier_params = {"C": np.linspace(0.01, 10, 100)}

    elif model_type == "SVC_linear":
        algo = SVC
        classifier_options = {
            "kernel": "linear",
            "probability": True,
            "random_state": random_state,
            "cache_size": 1e4,
            "max_iter": 1e6,
            "class_weight": "balanced",
        }
        classifier_params = {"C": np.linspace(10**-4, 10**4, 10)}
    elif model_type == "SVC_RBF":
        algo = SVC
        classifier_options = {
            "kernel": "rbf",
            "probability": True,
            "random_state": random_state,
            "cache_size": 1e4,
            "max_iter": 1e6,
            "class_weight": "balanced",
        }
        classifier_params = {
            "C": np.linspace(10**-4, 10**4, 10),
            "gamma": np.linspace(10**-4, 10**4, 10),
        }
    elif model_type == "RF":
        algo = RandomForestClassifier
        classifier_options = {
            "random_state": random_state,
            "class_weight": "balanced",
        }  #'max_features': "sqrt"
        classifier_params = {
            "n_estimators": [25, 50, 75, 100],
            # 'min_samples_leaf': [0.01, 0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }
    elif model_type == "DTC":
        algo = DecisionTreeClassifier
        classifier_options = {
            "random_state": random_state,
            "class_weight": "balanced",
        }  #'max_features': "sqrt"
        classifier_params = {
            "min_samples_leaf": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }

    elif model_type == "Nnet":
        algo = MLPClassifier
        classifier_options = {"max_iter": int(1e6), "random_state": random_state}
        classifier_params = {
            "hidden_layer_sizes": [2, 4, 10, 20],
            "alpha": np.linspace(1e-4, 1e3, 10),
            "learning_rate_init": np.linspace(1e-4, 1e-2, 10),
        }

    elif model_type == "GBC":
        algo = GradientBoostingClassifier
        classifier_options = {"random_state": random_state}
        classifier_params = {
            "n_estimators": [100, 200, 500],
            "min_samples_leaf": [0.01, 0.02, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }

    elif model_type == "XGB":
        algo = XGBClassifier
        classifier_options = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "learning_rate": 0.05,
            "scale_pos_weight": 5.0,
            "missing": np.nan,
            "random_state": random_state,
        }
        classifier_params = {
            "max_depth": [2, 3],
            "colsample_bytree": [0.2, 0.3, 0.4],
            "n_estimators": [15, 25, 50],
            "subsample": [0.8, 0.9],
            "colsample_bynode": [0.7, 0.8],
            "reg_lambda": [1e-05, 1.0],
        }

    name_suffix = "%s" % (model_type)
    model_info = {"model_name": model_type}

    pipeline_dict = {}
    params_dict = {"classifier": classifier_params}
    pipeline_options = {"classifier": classifier_options}

    if use_scale:
        pipeline_dict.update({"scale": scaler})

    pipeline_dict.update({"classifier": algo})
    gridsearch_dict.update(
        {
            "%s"
            % (name_suffix): {
                "pipeline_dict": pipeline_dict,
                "params_dict": params_dict,
                "pipeline_options": pipeline_options,
                "model_info": model_info,
            }
        }
    )

    return gridsearch_dict


def statistical_pipeline(
    X,
    y,
    pipeline_dict,
    params_dict,
    pipeline_options,
    random_state=111,
    n_jobs=None,
    n_splits_outer=5,
    n_splits_inner=3,
    n_repeats_outer=10,
    n_repeats_inner=10,
    metric="f1_score",
    refit_outer=True,
    verbose=0,
    refit_method=True,
):  #  refit_method = get_best_index

    # === NestedCV ===
    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits_outer, n_repeats=n_repeats_outer, random_state=random_state
    )
    inner_cv = RepeatedStratifiedKFold(
        n_splits=n_splits_inner, n_repeats=n_repeats_inner, random_state=random_state
    )

    print("Performing GridSearch")
    clf = GridSearchNestedCV(
        pipeline_dict,
        params_dict,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        n_jobs=n_jobs,
        pipeline_options=pipeline_options,
        metric=metric,
        verbose=verbose,
        refit_outer=refit_outer,
        return_train_score=True,
        imblearn_pipeline=True,
        refit_inner=refit_method,
        error_score=np.nan,
    )
    clf.fit(X, y)

    return clf
