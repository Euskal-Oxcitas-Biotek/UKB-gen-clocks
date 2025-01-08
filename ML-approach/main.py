# main.py

import preparation as prep #to create the sploit
import linear_model as cl #to train the (un)penalized linear models-based brain clocks
import nonlinear_model as ncl #to train the (un)penalized linear models-based brain clocks
import test_metrics_all as tm #to compute performance metrics in test set
import corrector as cr #to correct models not clearing t-test/p-values for all coefficients
import os
import shutil
import pickle
import pandas as pd
import json



def main(age_limits = [55,85], delta_exclusion_months = None, stratified_by_CN = True, time_splits = [0,5,10], correct_pvals = 1,
         run_id = 1, generate_test = True, delete_existing_test = True):
    """
    :param age_limits: Age interval in years (list: [a,b]) denoting the age bracket considered during training/testing of developed models. Default: a=55,b=85.
    :param delta_exclusion_months: MRI delta interval/list in months, (a,b]. Default: a=2, b=4.
    :param stratified_by_CN: Whether or not CN_type must be considered in stratified splitting. If not, just consider 'healthy_orx'. Default: True.
    :param database_train: Databases that must be fully considered in training set. Default: [].
    :param database_test: Databases that must be fully considered in test set. Default: [].
    :param time_splits: Time splits to be considered to generate CN_type. Default: [0,5,10].
    :param correct_pvals: Whether or not to correct models failing p-values (default p-val threshold: 0.05) and amount of repetitions. If 0, it is not performed,
           if more than 0, repeated as many times as correct_pvals. Default: 1.
    :param run_id: Execution id number which is used to look into corresponding history of settings combinations explored/corrected file. Default: 1.
    :param generate_test: Boolean, whether or not to compute test metrics. Default: True.
    :param delete_existing_test: Boolean, whether or not to eliminate and redo full test. Default: True
    """
    #-------------------------------------------------------------------------------------------------------------------
    # STEP -1: Delete preexisting test information
    if delete_existing_test:
        file_path_register = 'historic_register_' + str(run_id) + '.json'
        if not os.path.exists(file_path_register):
            examined_combinations = []
            corrected_ids = []
            data = {}
            id = 1
            data['examined_combinations'] = examined_combinations
            data['corrected_ids'] = corrected_ids
            data['tested'] = {}
            data['id'] = id
            with open(file_path_register, 'w') as file:
                json.dump(data, file)
        else:
            with open(file_path_register, 'r') as file:
                data = json.load(file)
            if len(data['tested'])>0:
                # Eliminate tested cases
                data['tested'] = {}
                with open(file_path_register, 'w') as file:
                    json.dump(data, file)
                print(file_path_register + "file updated")
                # Eliminate test directories
                directories_test = ['metrics_test/', 'plots/', 'pareto/']
                for directory in directories_test:
                    if os.path.exists(directory) and os.path.isdir(directory):
                        shutil.rmtree(directory)
                        print(f"Directory '{directory}' has been deleted.")
                    else:
                        print(f"Directory '{directory}' does not exist.")
    # ---------------------------------------------------------------------------------------------------------------
    # STEP 0: Check whether we already have a split available for the requested population and settings to analyze.
    preprocessing_type = 'age_' + str(age_limits) + '_time_splits_' + str(time_splits) + '_delta_exclusion_' + str(delta_exclusion_months)

    if os.path.isfile('data/' + 'train_data:' + preprocessing_type + '.csv'):
        train_data = pd.read_csv('data/' + 'train_data:' + preprocessing_type + '.csv')
        test_data = pd.read_csv('data/' + 'test_data:' + preprocessing_type + '.csv')
        test_data['education_years'] = pd.cut(test_data['education_years'], bins=[0, 7, 15, 30], labels=['0-7', '8-15', '16-30'],
                                       include_lowest=True, right=True)
        with open('data/binary_cases' + preprocessing_type + '.pkl', 'rb') as file:
            full_set_of_predictors, binary_cases_train, binary_cases_test, binary_columns_train, binary_columns_test = pickle.load(file)

    else:
        dp = prep.dataset_selection(age_limits = age_limits, time_splits = time_splits, delta_exclusion_months = delta_exclusion_months)
        train_data = dp.train_data
        test_data = dp.test_data
        full_set_of_predictors = dp.full_set_of_predictors
        binary_cases_train = dp.binary_cases_train
        binary_cases_test = dp.binary_cases_test
        binary_columns_train = dp.binary_columns_train
        binary_columns_test = dp.binary_columns_test
        preprocessing_type = dp.preprocessing_type
    print('** STEP 0 (preprocessing) finished **')
    # ---------------------------------------------------------------------------------------------------------------
    # STEP 1: Check whether there are already some pre-trained settings in corresponding .json file and exclude them from analysis, run remaining cases.

    for T1W in [{'ctx_feats': 'Merged', 'LR_feats': 'Merged', 'CC_feats': True, 'extra_feats': True}, {'ctx_feats': 'All', 'LR_feats': 'All', 'CC_feats': True, 'extra_feats': True}]:
        for training_population in ['orx','cole']:
            for gender in ['All', 'Male', 'Female']:
                for high_pairwise_correlation_removal in [True]:#, False]:
                    for method in ['ols','lasso', 'ridge', 'elastic_net', 'xgb_hyperopt', 'lgb_hyperopt','tpot', 'flaml']:
                        for sampling in [None, 'resample']:#, 'smote', 'adasyn']:
                            if method == 'tpot':
                                for eval_metric in ['neg_mean_absolute_error','neg_mean_squared_error']:
                                    model = ncl.LMB(age_limits = age_limits, gender=gender, training_population = training_population,
                                                   method = method, sampling = sampling, database_origin = 'data/' + 'train_data:' + preprocessing_type + '.csv',
                                                   T1W = T1W, evaluation_metric = eval_metric,
                                                   run_id = run_id, high_pairwise_correlation_removal = high_pairwise_correlation_removal)
                                    if model.proceed:
                                        model.fit(train_data)
                                    else:
                                        print('pre-executed case', model.combination)
                            elif method == 'flaml':
                                for eval_metric in ['mae','mse']:
                                    model = ncl.LMB(age_limits = age_limits, gender=gender, training_population = training_population,
                                                   method = method, sampling = sampling, database_origin = 'data/' + 'train_data:' + preprocessing_type + '.csv',
                                                   T1W = T1W, evaluation_metric = eval_metric,
                                                   run_id = run_id, high_pairwise_correlation_removal = high_pairwise_correlation_removal)
                                    if model.proceed:
                                        model.fit(train_data)
                                    else:
                                        print('pre-executed case', model.combination)
                            elif method == 'xgb_hyperopt':
                                for eval_metric in ['mae','mse']:
                                    print(eval_metric)
                                    model = ncl.LMB(age_limits = age_limits, gender=gender, training_population = training_population,
                                                   method = method, sampling = sampling, database_origin = 'data/' + 'train_data:' + preprocessing_type + '.csv',
                                                   T1W = T1W, evaluation_metric = eval_metric,
                                                   run_id = run_id, high_pairwise_correlation_removal = high_pairwise_correlation_removal)
                                    if model.proceed:
                                        model.fit(train_data)
                                    else:
                                        print('pre-executed case', model.combination)
                            elif method == 'lgb_hyperopt':
                                for eval_metric in ['mae','mse']:
                                    model = ncl.LMB(age_limits = age_limits, gender=gender, training_population = training_population,
                                                   method = method, sampling = sampling, database_origin = 'data/' + 'train_data:' + preprocessing_type + '.csv',
                                                   T1W = T1W, evaluation_metric = eval_metric,
                                                   run_id = run_id, high_pairwise_correlation_removal = high_pairwise_correlation_removal)
                                    if model.proceed:
                                        model.fit(train_data)
                                    else:
                                        print('pre-executed case', model.combination)
                            else:
                                model = cl.LMB(age_limits = age_limits, gender=gender, training_population = training_population,
                                               method = method, sampling = sampling, database_origin = 'data/' + 'train_data:' + preprocessing_type + '.csv',
                                               T1W = T1W,
                                               run_id = run_id, high_pairwise_correlation_removal = high_pairwise_correlation_removal)
                                if model.proceed:
                                    model.fit(train_data)
                                else:
                                    print('pre-executed case', model.combination)
    print('** STEP 1 (model training) finished')
    # ---------------------------------------------------------------------------------------------------------------
    # STEP 2: Correct those models failing the individualized coefficient t-test if requested by user (this step can repeated)
    for i in range(correct_pvals):
        model = cr.retrain(X = train_data, run_id = run_id)
        model.models_to_correct()
        print('** STEP 2 (model correction, repetition ' + str(i+1) + ') finished:' + str(model.amount_corrected) + ' models corrected.')
    # ---------------------------------------------------------------------------------------------------------------
    # STEP 3: Model testing and selection
    if generate_test:
        for db in ['UKBB', 'ADNI', 'NACC', None]:
            for gender_test in ['Male', 'Female']:
                for include_age_biased_models in [True, False]:
                    for p_value_elimination in [True]: #True Get the various Pareto fronts selecting or not models that clear t-test (for coefficients p-value)
                      output = tm.output_generator(test_data, binary_cases_test, binary_columns_test, age_limits = age_limits, predictors_test = full_set_of_predictors,
                                                   gender_test=gender_test, save_analyzed_cases = True, p_value_elimination = p_value_elimination, include_age_biased_models = include_age_biased_models,
                                                   pareto_feature_cases = [['MAE', 'MAE_max_bin','auroc_CN_noCN_orx'], ['MAE_bounded', 'MAE_max_bin_bounded','auroc_CN_noCN_orx_bounded']],
                                                   baseline_test_feats_set = [['age_at_scan'],['age_at_scan','MOCA', 'MMSE', 'CDR'],['age_at_scan', 'CDR'],['age_at_scan', 'MOCA'],
                                                                              ['age_at_scan', 'MMSE']],
                                                   age_bound = [55,80], db = db)#,'MOCA', 'MMSE', 'CDR'])
                      output.test_models()
        print('** STEP 3 (model testing/selection finished')

if __name__ == "__main__":
    main(age_limits = [55,85], correct_pvals = 10, generate_test = True, delete_existing_test= True)
