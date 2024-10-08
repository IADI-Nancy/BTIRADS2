# BTI-RADS 2.0

This repository contains the source code for the article "Enhanced Focal Bone Tumor Classification with Machine Learning-Based Stratification: A Multicenter Retrospective Study". 

The research tool can be accessed via https://bti-rads.cic-it-nancy.fr.


## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

The pipeline requires individual excel files for training and test sets and assumes target labels in a "Label" column. To split your data into train/test, run:

```bash
python main.py split --data_dir <input_file> --outputs_dir <outputs_dir> --datasplit_ratio <datasplit_ratio> 
```

where `<input_file>` is the path to your global database excel file, `<outputs_dir>` is the output directory where the files 'data_train.xlsx' and 'data_test.xlsx' will be saved, and `<datasplit_ratio>` specifies the split ratio. Expected feature (excel column) names can be found in `format_utils.py`

To train a classifier ensemble using nested cross-validation (including prior feature selection), run:

```bash
python main.py train --data_dir <input_file> --outputs_dir <outputs_dir> --modality <modality> --fs_aggregation
```

where `<input_file>` is the path to the formatted training data input excel file (e.g. 'data_train.xlsx'), `<outputs_dir>` is the output directory where the model checkpoints will be saved, and `<modality>` is the imaging modality of interest (multimodal, rx, or mr). Note that you might want to adapt some of the parameter settings that are currently hardcoded in `train.py` (e.g. number of features, number of inner and outer folds,...). The argument `--fs_aggregation` results in bootstrapped ranking aggregation during feature selection. The number of desired features can be modified by providing `--fs_nfeat_list`.


To fit BTI-RADS thresholds that split the model output space into four risk categories, run:

```bash
python main.py btirads2 --data_dir <input_file> --outputs_dir <outputs_dir> --modality <modality> 
```
where `<input_file>` is the path to the formatted training data input excel file (e.g. 'data_train.xlsx'), `<outputs_dir>` is the output directory where the thresholds and BTI-RADS scores will be saved.


To evaluate the nested cross-validation performance for the outer fold validation sets, run:

```bash
python main.py evaluate --evaluate_cv --data_dir <input_file> --outputs_dir <outputs_dir> --modalities <modalities>
```
where `<input_file>` is the path to the formatted training data input excel file (e.g. 'data_train.xlsx'), `<outputs_dir>` is the output directory where the performance scores will be saved, and  `--modalities` the list of modalities to consider during evaluation.


To evaluate the ensemble model on the held-out test set, run:

```bash
python main.py evaluate --evaluate_test --evaluate_btirads --data_dir <input_file> --outputs_dir <outputs_dir>  --modalities <modalities>
```
where `<input_file>` is the path to the formatted test data input excel file (e.g. 'data_test.xlsx'), `<outputs_dir>` is the output directory where the performance scores will be saved. The argument `--evaluate_btirads` furthermore provides BTI-RADS scores for the test samples and test cohort malignancy frequencies for each BTI-RADS category.


To visualize shap values for a selection of samples for a specific setup, run:

```bash
python main.py shap_analysis --data_dir <input_file> --outputs_dir <outputs_dir> --modality <modality> --fs_name <fs> --sample_list <samples>
```
where `<input_file>` is the path to the formatted data input excel file (e.g. 'data_test.xlsx'), `<outputs_dir>` is the output directory where the shap plots will be saved, `<modality>` is the imaging modality of interest (multimodal, rx, or mr), `<fs>` the feature selection setup (e.g. 'chi2_importance_score_27'), and `<samples>` the list of patient indices. 

## References

The nested cross-validation and feature selection pipelines were adapted from the following public repository with MIT license: https://github.com/IADI-Nancy/Sklearn_NestedCV. 
