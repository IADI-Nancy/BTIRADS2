import argparse
from typing import Sequence, Optional


def build_parser_common():
    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("common")

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Level of verbosity: (0: warning/error, 1: info, 2: detailed info)",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=11111,
        help="Random seed",
    )

    parser.add_argument(
        "--cpu_range", type=int, nargs="+", default=[0, 5], help="CPU core range"
    )

    return parser


def build_parser_inputs():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("inputs")

    parser.add_argument(
        "--data_dir",
        default="./data/data_test.xlsm",
        type=str,
        help="Dataset excel file.",
    )

    return parser


def build_parser_json():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("json")

    parser.add_argument(
        "--json",
        type=str,
        help="Json buffer string.",
    )

    return parser


def build_parser_outputs():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("outputs")

    parser.add_argument(
        "--outputs_dir",
        default="./outputs",
        type=str,
        help="Output directory.",
    )

    parser.add_argument(
        "--filename_prefix",
        default="",
        type=str,
        help="Whether to append a prefix to the default output filenames.",
    )
    return parser


def build_parser_datasplit():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("datasplit")

    parser.add_argument(
        "--datasplit_ratio",
        default=0.2,
        type=float,
        help="Split data intro train:test data with ratio defining the percentage of test samples.",
    )

    return parser


def build_parser_modality():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("modality")

    parser.add_argument(
        "--modality",
        default="multimodal",
        type=str,
        choices=["multimodal", "rx", "mr"],
        help="Choice of imaging modality, one of: multimodal, rx, mr.",
    )

    return parser


def build_parser_model_train():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("training")

    parser.add_argument(
        "--feature_selection_type",
        default="chi2",
        type=str,
        choices=["chi2"],
        help="Feature selection algorithm.",
    )

    parser.add_argument(
        "--model_type",
        default="XGB",
        type=str,
        choices=["XGB"],
        help="Classifier type.",
    )

    parser.add_argument(
        "--fs_aggregation",
        action="store_true",
        help="Bootstrap agggreation during training",
    )

    parser.add_argument(
        "--fs_only",
        action="store_true",
        help="Perform only feature selection",
    )

    parser.add_argument(
        "--fs_nfeat_list",
        nargs="+",
        default=[10, 27],
        type=int,
        help="Number of selected features.",
    )

    return parser


def build_parser_model_test():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("test")

    parser.add_argument(
        "--shap_plots",
        action="store_true",
        help="Are shap plots desired?",
    )

    return parser


def build_parser_evaluation():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("evaluation")

    parser.add_argument(
        "--evaluate_cv",
        action="store_true",
        help="Do you wish for nested cross-validation performance evaluation?.",
    )

    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        help="Do you wish for held-out test set performance evaluation?.",
    )

    parser.add_argument(
        "--evaluate_btirads",
        action="store_true",
        help="Do you wish for held-out test set btirads classification performance evaluation?.",
    )

    parser.add_argument(
        "--n_bootstrap",
        default=200,
        type=int,
        help="Number of bootstrap samples used for confidence intervals.",
    )
    parser.add_argument(
        "--ci",
        nargs="+",
        default=[0.025, 0.975],
        type=int,
        help="Lower and upper percentile for (95%) confidence intervals.",
    )
    parser.add_argument(
        "--ref_model",
        default="multimodal_chi2_importance_score_27_XGB",
        type=str,
        help="Reference model.",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["multimodal", "rx", "mr"],
        type=str,
        help="modalities to consider in evaluation",
    )

    parser.add_argument(
        "--modality_txts",
        nargs="+",
        default=["multimodal, ", "CT, ", "MRI, "],
        type=str,
        help="modality output prefixes",
    )

    return parser


def build_parser_model_setup():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("setup")

    parser.add_argument(
        "--fs_name",
        default="chi2_importance_score_27",
        type=str,
        help="Feature selection setup to evaluate.",
    )
    parser.add_argument(
        "--model_type",
        default="XGB",
        type=str,
        choices=["XGB"],
        help="Classifier type.",
    )

    return parser


def build_parser_model_setups():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("setup")

    parser.add_argument(
        "--fs_names_list",
        nargs="+",
        default=["chi2_importance_score_10", "chi2_importance_score_27"],
        type=str,
        help="Feature selection setups to evaluate.",
    )
    parser.add_argument(
        "--model_type",
        default="XGB",
        type=str,
        choices=["XGB"],
        help="Classifier type.",
    )

    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="Decision threshold.",
    )

    return parser


def build_parser_reference_setup():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("reference_setup")

    parser.add_argument(
        "--reference",
        default="multimodal_chi2_importance_score_27_XGB",
        type=str,
        help="Reference setup used for model comparison (stats tests).",
    )

    return parser


def build_parser_shap():

    _parser = argparse.ArgumentParser(add_help=False)

    parser = _parser.add_argument_group("shap")

    parser.add_argument(
        "--sample_list",
        nargs="+",
        default=[0],
        type=int,
        help="Samples for which shap plots are generated.",
    )

    return parser


def add_subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    help: Optional[str],
    description: Optional[str],
    parents: Sequence,
) -> argparse.ArgumentParser:

    parser = subparsers.add_parser(
        name=name,
        help=help,
        description=description,
        parents=parents,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    return parser


def build_command_split(subparsers):
    parser = add_subcommand(
        subparsers,
        name="split",
        help="Perform train:test datasplit",
        description=("Define a datasplit into exploratory and test set. "),
        parents=[
            build_parser_inputs(),
            build_parser_outputs(),
            build_parser_datasplit(),
            build_parser_common(),
        ],
    )
    return parser


def build_command_train(subparsers):
    parser = add_subcommand(
        subparsers,
        name="train",
        help="Train a model using nested cross-validation",
        description=(
            "Optimize a binary classifier and BTI-RADS algorithm for lesion malignancy classification. "
            "This command can be used to perform feature selection and nested cross-validation. "
        ),
        parents=[
            build_parser_inputs(),
            build_parser_modality(),
            build_parser_outputs(),
            build_parser_model_train(),
            build_parser_common(),
        ],
    )
    return parser


def build_command_btirads2(subparsers):
    parser = add_subcommand(
        subparsers,
        name="btirads2",
        help="Fit thresholds for BTIRADS2 classification",
        description=(
            "Optimize probability thresholds for multiclass risk stratification on the exploratory dataset. "
        ),
        parents=[
            build_parser_inputs(),
            build_parser_modality(),
            build_parser_outputs(),
            build_parser_evaluation(),
            build_parser_model_setups(),
            build_parser_common(),
        ],
    )
    return parser


def build_command_evaluate(subparsers):
    parser = add_subcommand(
        subparsers,
        name="evaluate",
        help="Evaluate the model(s)",
        description=(
            "Run performance assessment for one or multiple models. "
            "This command can be used to compute evaluation scores such as F1score/accuracy, and to compare performances. "
        ),
        parents=[
            build_parser_inputs(),
            build_parser_modality(),
            build_parser_outputs(),
            build_parser_evaluation(),
            build_parser_model_setups(),
            build_parser_common(),
        ],
    )
    return parser


def build_command_shap_analysis(subparsers):
    parser = add_subcommand(
        subparsers,
        name="shap_analysis",
        help="Perform shap value analysis",
        description=("Visualization of decision paths and feature importance"),
        parents=[
            build_parser_inputs(),
            build_parser_modality(),
            build_parser_shap(),
            build_parser_model_setup(),
            build_parser_outputs(),
            build_parser_common(),
        ],
    )
    return parser


def main_parser():

    parser = argparse.ArgumentParser(
        prog="BTI-RADS2.0",
        description="Bone tumor imaging reporting and data system 2.0",
        epilog="Run 'main.py --command --help' for more information on a command.\n",
    )

    parser.add_argument(
        "--command",
        type=str,
        choices=["split", "train", "btirads2", "evaluate", "shap_analysis"],
    )

    subparsers = parser.add_subparsers(
        title="commands", metavar="command", dest="command"
    )

    build_command_split(subparsers)
    build_command_train(subparsers)
    build_command_btirads2(subparsers)
    build_command_evaluate(subparsers)
    build_command_shap_analysis(subparsers)

    return parser, subparsers
