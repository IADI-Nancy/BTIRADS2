import os
import joblib
import numpy as np
import pandas as pd
from .format_utils import (
    features_multimodal,
    features_mr,
    features_rx,
    features_numeric,
)
from sklearn.preprocessing import OneHotEncoder


def preprocess_pipeline(data, modality, mod_outputs_dir, is_fitted=True):

    # feature selection
    data, features_cat, features_num = feature_selection(data, type=modality)

    if is_fitted:
        # categorical feature processing: using fitted one-hot encoder
        encoder = joblib.load(os.path.join(mod_outputs_dir, "onehot_encoder.joblib"))
        data, _ = feature_encoding(data, features_cat, features_num, enc=encoder)
    else:
        # categorical feature processing: define one-hot encoder and save to joblib object
        data, encoder = feature_encoding(data, features_cat, features_num)
        joblib.dump(encoder, os.path.join(mod_outputs_dir, "onehot_encoder.joblib"))

    # drop some encoded features
    data_with_missing = data.copy()
    data = drop_unknown_na_and_binary(data)

    if is_fitted:
        # numerical feature processing: percentile clipping, save thresholds to file
        percentiles = np.load(
            os.path.join(mod_outputs_dir, "percentiles.npy"), allow_pickle=True
        ).item()
        data, _ = preprocess_numeric(data, features_num, percentiles=percentiles)
    else:
        # numerical feature processing: percentile clipping, save thresholds to file
        data, percentiles = preprocess_numeric(data, features_num)
        np.save(os.path.join(mod_outputs_dir, "percentiles.npy"), percentiles)

    # Set categories of features with missing values (encoded as "unknown" category) to np.nan
    data_nanformat = data.copy()

    missing_data_cols = [c for c in data_with_missing.columns if "_unknown" in c]
    for c in missing_data_cols:
        feat = c.split("_unknown")[0]
        associated_cols = [
            ca for ca in data_with_missing.columns if feat == ca.split("_")[0]
        ]

        for ca in associated_cols:
            if ca in data_nanformat.columns:
                data_nanformat.loc[data_with_missing[c] == 1, ca] = np.nan

    return data, data_nanformat, features_num, features_cat


def feature_selection(data, type="mulimodal"):

    if type == "mr":
        features_selected = features_mr

    elif type == "rx":
        features_selected = features_rx

    elif type == "multimodal":
        features_selected = features_multimodal
    else:
        raise "Modality type not supported"

    data_out = data.copy()
    for c in data.columns:
        if c not in features_selected:
            data_out.drop(c, axis=1, inplace=True)

    features_num = [f for f in features_numeric if f in features_selected]
    features_cat = [f for f in features_selected if f not in features_num]

    return data_out, features_cat, features_num


def feature_encoding(data, features_cat, features_num, enc=None):

    data_num = data[features_num]

    if enc is None:
        enc = OneHotEncoder(
            categories="auto", dtype=np.float64, sparse=False, handle_unknown="ignore"
        ).fit(data[features_cat])

        # as some features might not present missing values for the training set
        # but for the test set, add an unknown category for all features
        categories_with_unknown = enc.categories_
        for f_i, f_cat in enumerate(categories_with_unknown):

            if "unknown" not in f_cat:
                categories_with_unknown[f_i] = np.append(
                    categories_with_unknown[f_i], "unknown"
                )

        enc = OneHotEncoder(
            categories=categories_with_unknown,
            dtype=np.float64,
            sparse=False,
            handle_unknown="ignore",
        ).fit(data[features_cat])

    data_enc = pd.DataFrame(
        columns=enc.get_feature_names_out(),
        data=enc.transform(data[features_cat]),
        index=data_num.index,
    )
    data = pd.concat([data_num, data_enc], axis=1)

    return data, enc


def drop(data, excluded_features):

    for f in data.columns:
        if f in excluded_features:
            data = data.drop(f, axis=1)

    return data


def drop_unknown_na_and_binary(data):

    excluded_features = []

    # exclude missing data category
    excluded_unknown = [f for f in data.columns if "unknown" in f]
    excluded_features = excluded_features + excluded_unknown
    data = drop(data, excluded_unknown)

    # exclude NA category except for the expandability feature
    excluded_na = [
        f
        for f in data.columns
        if (
            (f.split("_")[-1] == "not applicable")
            and not f == "Expandability_not applicable"
        )
    ]
    excluded_features = excluded_features + excluded_na
    data = drop(data, excluded_na)

    # for all binary features, drop duplicate (preferable the "absence" category)
    excluded_binary = []
    excluded_binary_base = []
    for fname in data.columns.values:
        if "_" in fname:
            fname_base = fname.split("_")[0]
            # if not already excluded
            if not fname_base in excluded_binary_base:
                f_cats = [
                    fname2
                    for fname2 in data.columns.values
                    if (
                        (fname2.split("_")[0] == fname_base)
                        and not ("not applicable" in fname2)
                    )
                ]
                n_cats = len(f_cats)

                if n_cats == 2:
                    if f_cats[0].split("_")[-1] == "absence":
                        excluded_binary.append(f_cats[0])
                    else:
                        excluded_binary.append(f_cats[1])
                    excluded_binary_base.append(fname_base)

    excluded_features = excluded_features + excluded_binary
    data = drop(data, excluded_binary)

    # print("Excluded features: ", excluded_features)

    return data


def preprocess_numeric(data, features_num, percentiles=None):

    if percentiles is None:
        percentiles = {}
        get_percentiles = True
    else:
        get_percentiles = False

    for f in features_num:
        if get_percentiles:
            # default percentile computation uses method='linear' if percentile is intermediate position
            percentiles[f] = np.array(
                [np.percentile(data[f], 5), np.percentile(data[f], 95)]
            )
        data[f].clip(lower=percentiles[f][0], upper=percentiles[f][1], inplace=True)

    return data, percentiles
