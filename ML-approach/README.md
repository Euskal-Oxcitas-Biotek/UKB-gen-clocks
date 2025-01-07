# ML MODELS TRAINING/TESTING IMPLEMENTATION
Initial pipeline implementation for model development (including age bias correction) for UKBB paper. Please refer to main.py to check configuration and 
specific steps followed in this process. By running this file linear and non-linear models are rained, tested and selected (used metadata files are not made public due MTA agreement of UK Biobank, ADNI and NACC)

In this repo, an initial pipeline implementation for model development (including age bias correction) for UKBB paper.
This approach is focused
on standard linear regression type of models (including OLS, LASSO, Ridge and Elastic-Nets) and non-linear approaches via TPOT and FLAML as well as various 
preprocessing choices. It also includes the automated computation of various age-bias correction methodologies
(including Cole's, Besheti's and Lange's). In here, we have the two following type of files:

1) 6 .py files: 1.1) main.py, which includes a comprehensive examples showing the use of the various functions
   presented in this repository. 1.2) Preparation.py, which is used to generate the data split using the entire metadata facilitated
   by SV on 21/06 (keeping UKBB as training/test and ADNI/NACC just for test). 1.2) lineal_model_AVIV.py/nonlineal_model_AVIV.py:
    In this file we have the classes that are used to train
   customized linear/non-linear regression brain clocks. 1.3) train_features.py: In here, we compute various metrics of the developed models over the training set, 
   we compute the p-values of the corresponding coefficients and provide additional info related to the optimizer
   used in the penalty-based models, 1.4) corrector.py: In here, we have the function used to select and correct those models for
   which we have coefficients failing the corresponding t-test to measure coefficient value reliability, 1.5) test_metrics_all.py:
   In here, we compute test metrics, generate the plots of interest and provide a model selection tool inspired in Pareto analysis.

2) Two additional .py files are available to reformat output generated in test_metrics_all.py for standardization against DL-based approaches. 
   
# Libraries Required

Just standard Python libraries are used in this project: Numpy, Pandas, Seaborn, Matplotlib, Scipy, Sklearn, Imblearn
For non-linear approaches, we use other libraries including flaml, tpot, hgboost, xgboost, lightgbm.
