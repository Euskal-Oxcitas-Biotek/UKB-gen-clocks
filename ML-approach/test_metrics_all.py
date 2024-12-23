import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import linear_model_AVIV as cl
import nonlinear_model_AVIV as ncl
from sklearn.metrics import roc_auc_score
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import os
import ast
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


class output_generator():
    def __init__(self, test_set, binary_cases_test, binary_columns_test, gender_test='Male',
                 age_limits=[55, 85], bins_length=5, predictors_test=[], directory_metrics_train='metrics_train/',
                 directory_metrics='metrics_test', directory_plots='plots', directory_pareto='pareto',
                 directory_models='models/', save_plots=True, p_value_elimination=False,
                 pareto_feature_cases=[['MAE_full', 'MAE_max_bin', 'auroc_CN_noCN']],
                 save_analyzed_cases=False, event_limit_cases=50, test_name='all_T1w_feats',
                 include_age_biased_models=True, run_id=1, baseline_test_feats_set=[['age_at_scan']], age_bound = [55,80], db = None):
        """
        :param test_set: Test set to validate chronological age prediction capacity of the model.
        :param binary_columns_test: List contaning the full set of binary problems (column names in dataframe) that could be potentially studied.
        :param binary_cases_test: Amount of cases available in each class for all binary problems that could be studied.
        :param predictors_test: The set of volumetric features of the models to be analyzed/compared in the test set. Default: [].
        :param gender_test: Gender to be considered when testing the models. Options: 'Male', 'Female', 'All'. Default: 'Male'.
        :param bins_length: Length of bins (in years) to be used to compute the test performance metrics. Default: 5.
        :param directory_plots: Directory where generated plots should be stored. Default: 'plots/'+'_'+gender_test+'_'+predictors.
        :param directory_pareto: Directory where generated plots should be stored. Default: 'pareto/'+'_'+gender_test+'_'+predictors.
        :param directory_metrics_train: Directory where train metrics are stored. Default: 'metrics_train'
        :param directory_metrics: Directory where test metrics should be stored. Default: 'metrics_test/'+'_'+gender_test+'_'+predictors.
        :param directory_models: Directory where generated models should be stored. Default: 'metrics_models/'.
        :param save_plots: Boolean (True, False) to decide whether or not to save/generate any plots. Default: True
        :param p_value_elimination: Boolean (True, False) to decide whther or not to neglect from the analysis models that do not clear p-value test. Default: False
        :param pareto_feature_cases: Test metric set of features to be considered when determining Pareto fronts.
        :param save_analyzed_cases: Boolean (True, False) to decide whether or not to save in a .csv file all models prior Pareto selection. Default: False
        :param test_name: Additional name (string) to distinguish the associated folder naming with results of this test. Default: 'all_T1w_feats'
        :param include_age_biased_models: Boolean to decide whether or not to allow inclusion of age-biased models among selected models. Default: True
        :param run_id: Execution id number which is used to look into corresponding history of settings combinations explored/corrected file. Default: 1.
        :param baseline_test_feats_set: Features, no-MRI related, that will be used to compare disease prediction vs brain clocks from Pareto, e.g.,
               [['CDR','MMSE','MOCA','age_at_scan']]. Default: [['age_at_scan']].
        :param age_bound: If set to not None it is the maximum age considered if we want to recompute the MAE in the group [age_limits[0], age_bound]
        :param db: Individual database to be considered in the analysis. Default: None (consider the full batch).
        """

        # -------------------------------------------------------------------------------
        # STEP 1: Select the set of features that contain all of the features that will
        #         considered across all models to be compared and create the bins limits to be used

        self.age_limits = age_limits
        self.age_bound = age_bound
        self.db = db
        self.include_age_biased_models = include_age_biased_models
        self.event_limit_cases = event_limit_cases
        self.predictors_test = predictors_test + [
            'PTGENDER']  # 'PTGENDER' is added for those models trained over both genders

        self.bin_edges = []
        current = self.age_limits[0]
        while current <= self.age_limits[1]:
            self.bin_edges.append(current)
            current += bins_length

        self.age_cases_to_analyze = ['']
        if self.age_bound is not None:
            self.age_cases_to_analyze.append('_bounded')

        self.gender_test = gender_test
        self.save_plots = save_plots
        self.save_analyzed_cases = save_analyzed_cases
        self.pareto_feature_cases = pareto_feature_cases
        # -------------------------------------------------------------------------------
        # STEP 2: Keep the cases of interest in the analysis (stratifying by gender and
        #         eliminating missing values)
        if self.gender_test != 'All':
            test_set = test_set[test_set['biological_sex'] == self.gender_test]
        predictors = self.predictors_test + ['age_at_scan']  # 'age_at_scan' is added as it is the target
        test_set = test_set.dropna(subset=predictors)
        if self.db is not None:
            test_set = test_set[test_set['db_name'] == self.db]
        self.X_test = test_set.reset_index(drop=True)
        self.binary_cases_test = binary_cases_test
        self.binary_columns_test = binary_columns_test
        self.p_value_elimination = p_value_elimination
        self.baseline_test_feats_set = baseline_test_feats_set
        # ------------------------------------------------------------------------------
        # STEP 3: Create directories to save results if do not exist already
        if self.p_value_elimination:
            added = '_pval'
        else:
            added = ''

        if self.include_age_biased_models:
            added += '_age-biased'

        self.test_name = str(self.db) + '_' + gender_test + '_' + test_name + added
        self.directory_metrics = directory_metrics + '/' + self.test_name + '/'
        self.directory_plots = directory_plots + '/' + self.test_name + '/'
        self.directory_pareto = directory_pareto + '/' + self.test_name + '/'
        self.directory_metrics_train = directory_metrics_train
        self.directory_models = directory_models

        if not os.path.exists(self.directory_metrics):
            os.makedirs(self.directory_metrics, exist_ok=True)

        if not os.path.exists(self.directory_pareto):
            os.makedirs(self.directory_pareto, exist_ok=True)

        if self.save_plots:
            if not os.path.exists(self.directory_plots):
                os.makedirs(self.directory_plots, exist_ok=True)

        # ---------------------------------------------------------------------------------------------------------------
        self.run_id = run_id
        self.historic_filepath = 'historic_register_' + str(
            self.run_id) + '.json'  # Extract history to only perform computations to have not been done previously
        with open(self.historic_filepath, 'r') as file:
            data_history = json.load(file)
            self.examined_combinations = data_history['examined_combinations']
            self.corrected_ids = data_history['corrected_ids']
            self.id = data_history['id']
            self.tested = data_history['tested']
            if self.test_name not in self.tested:
                self.tested[self.test_name] = []

    def test_models(self):
        """
        :return: Function to evaluate performance of the selected set of models in terms of age/disease prediction.
        """
        self.models_to_test()

        # -------------------------------------------------------------------------------
        # STEP 1: Get performance metrics comparing across all developed models for Pareto selection

        self.events = []
        for col in self.binary_columns_test:
            if min(self.binary_cases_test[col].values) > self.event_limit_cases:
                self.events.append(col)

        for idx, name in enumerate(self.cases_to_analyze['name']):
            if name not in self.tested[self.test_name]:
                pkl_name = self.cases_to_analyze['save_model'].iloc[idx]
                if self.cases_to_analyze['method'].iloc[idx] in ['tpot', 'flaml']:
                    self.model = ncl.LMB.load_model(self.directory_models + pkl_name + '.pkl')
                else:
                    self.model = cl.LMB.load_model(self.directory_models + pkl_name + '.pkl')
                self.bias_type = self.cases_to_analyze['age-bias'].iloc[idx]
                self.name = name
                self.modelling_strategy = self.cases_to_analyze['modelling_strategy'].iloc[idx]
                self.method = self.cases_to_analyze['method'].iloc[idx]
                self.training_population = self.cases_to_analyze['training_population'].iloc[idx]
                self.used_predictors = self.cases_to_analyze['non_null_predictors'].iloc[idx]
                self.amount_used_predictors = self.cases_to_analyze['non_null_preditors_set_size'].iloc[idx]
                self.sampling = self.cases_to_analyze['sampling'].iloc[idx]
                self.MAE_training = self.cases_to_analyze['MAE'].iloc[idx]
                self.MAE_group_training = self.cases_to_analyze['MAE_group'].iloc[idx]
                self.sex = self.cases_to_analyze['gender'].iloc[idx]
                self.predictors_origin = self.cases_to_analyze['predictors_origin'].iloc[idx]
                self.preprocessing = self.cases_to_analyze['preprocessing'].iloc[idx]
                self.eval_metric = self.cases_to_analyze['eval_metric'].iloc[idx]

                print('case:', self.name, '(', idx, '/', len(self.cases_to_analyze), ')')
                self.X_test['brain_age'] = self.model.predict(self.X_test, bias_type=self.bias_type,
                                                              y=self.X_test['age_at_scan'].values)
                self.X_test['PAD'] = self.X_test['brain_age'] - self.X_test['age_at_scan']
                self.X_test['abs_PAD'] = abs(self.X_test['PAD'])
                self.age_prediction_performance()
                self.disease_prediction_performance()

                self.tested[self.test_name].append(name)
                self.update_history()

        # -------------------------------------------------------------------------------
        # STEP 2: Extract models using Pareto strategy over preselected metrics and compare preselected
        #         models to baseline metrics on disease prediction
        for features_pareto in self.pareto_feature_cases:
            self.pareto_preselection(features_pareto=features_pareto)  # Pareto selection
            for features in self.baseline_test_feats_set:  # If baseline metrics are provided, extract largest subset of test set containing all selected baseline features
                self.baseline_test_feats = features
                self.X_test_feats = self.X_test.dropna(
                    subset=self.baseline_test_feats)  # Keep cases containing all baseline features of interest
                print('** Amount of cases to compare all baseline metrics VS Pareto-extracted models:' + str(
                    len(self.X_test_feats)) + '**')
                self.events_test_feats = []
                for col in self.binary_columns_test:
                    filtered_df = self.X_test_feats.dropna(subset=[col])
                    value_counts = filtered_df[col].value_counts()
                    if len(value_counts) > 1:
                        if value_counts.min() > self.event_limit_cases:
                            self.events_test_feats.append(col)
                print('** binary events available for baseline comparison: ', self.events_test_feats, ' **')
                if len(self.X_test_feats) > 0:
                    self.test_models_baseline_feats(
                        features_pareto=features_pareto)  # Compare performance of selected models to baseline metrics

    def models_to_test(self):
        """
        :return: Function to decide pre-trained models to be tested.
        """
        dataframes = []
        for filename in os.listdir(self.directory_metrics_train):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory_metrics_train, filename)
                df = pd.read_csv(file_path)
                if (self.include_age_biased_models) or (str(df['age-bias'].iloc[0])=='nan'):
                    if all(item in self.predictors_test for item in ast.literal_eval(df['predictors'].values[0])):
                        if (df['gender'].values[0] in ['All', self.gender_test]):
                            if self.p_value_elimination:
                                if df['p-values_success'].values[0]:
                                    dataframes.append(df)
                            else:
                                dataframes.append(df)

        unique_df = pd.concat(dataframes, ignore_index=True)
        self.cases_to_analyze = unique_df

    def age_prediction_performance(self, save_metrics=False):
        """
        :param save_metrics: Boolean (True, False) to decide wether or not to save the computed metrics for age predition performance. Default: False.
        :return: This function is used to compute various quality metrics and plots for the brain age as a predictor of chronological age on CN individuals.
        """

        if not save_metrics:
            df = pd.DataFrame([[self.name, self.modelling_strategy, self.method, self.training_population,
                                self.used_predictors, self.amount_used_predictors, self.sampling, self.sex, self.predictors_origin,
                                self.preprocessing, self.eval_metric, self.MAE_training, self.MAE_group_training]],
                              columns=['name', 'modelling_strategy', 'method', 'training_population', 'used_predictors',
                                       'used_predictors_set_size', 'sampling', 'gender', 'predictors_origin', 'preprocessing',
                                       'eval_metric', 'MAE_training', 'MAE_group_training'])
        else:
            df = self.metrics


        for case in self.age_cases_to_analyze:
            mae = {}
            mse = {}
            mae_bins = {}
            meanPAD_bins = {}
            r_squared = {}
            max_bin_MAE = {}
            groups_MAE = {}
            groups_stat_PAD = {}
            pearson_age_bio = {}
            pearson_age_PAD = {}

            for healthy in ['healthy_orx', 'healthy_cole']:  # Consider both alternatives for CN definition
                X_test = self.X_test[self.X_test[healthy]]
                if case == '_bounded':
                    X_test = X_test[(X_test['age_at_scan'] >= self.age_bound[0]) & (X_test['age_at_scan'] <= self.age_bound[1])]
                X_test = X_test.reset_index()
                # -------------------------------------------------------------------------------
                # STEP 0: MAE/MSE/R^2 computation per age bin and over full test set
                mse[healthy] = mean_squared_error(X_test['age_at_scan'], X_test['brain_age'])
                mae[healthy] = mean_absolute_error(X_test['age_at_scan'], X_test['brain_age'])
                r_squared[healthy] = r2_score(X_test['age_at_scan'], X_test['brain_age'])
                pearson_age_bio[healthy] = X_test['brain_age'].corr(X_test['age_at_scan'], method='pearson')
                pearson_age_PAD[healthy] = X_test['PAD'].corr(X_test['age_at_scan'], method='pearson')
                groups_MAE[healthy] = {}
                groups_stat_PAD[healthy] = {}
                mae_bins[healthy] = {}
                meanPAD_bins[healthy] = {}

                for i in range(len(self.bin_edges)-1):
                    bin_test = X_test[(X_test['age_at_scan'] >= self.bin_edges[i]) & (X_test['age_at_scan'] <= self.bin_edges[i+1])]
                    if len(bin_test) > 0:#bin_test['age_at_scan'].min() < bin_test['age_at_scan'].max():
                        mae_bins[healthy][
                            str(self.bin_edges[i]) + '-' + str(self.bin_edges[i+1])] = mean_absolute_error(
                            bin_test['age_at_scan'],
                            bin_test['brain_age'])
                        meanPAD_bins[healthy][
                            str(self.bin_edges[i]) + '-' + str(self.bin_edges[i+1])] = np.mean(bin_test['age_at_scan'] - bin_test['brain_age'])

                max_bin_MAE[healthy] = max(mae_bins[healthy].values())
                # -------------------------------------------------------------------------------
                # STEP 1: MAE per non-age related groups of interest. Add multigroup stat test on abs(PAD)
                for col in ['db_name', 'machine_model', 'manufacturer', 'CN_type', 'ethnicity','education_years']:
                    G = X_test.dropna(subset=[col]).copy()
                    G = G.reset_index(drop=True)
                    num_tests = len(G[col].unique())
                    groups_MAE[healthy][col] = {}
                    groups_stat_PAD[healthy][col] = {}
                    normality_p_values = []
                    for origin in G[col].unique():
                        indices = G[G[col] == origin].index
                        groups_MAE[healthy][col][origin] = mean_absolute_error(G['age_at_scan'].iloc[indices],
                                                                   G['brain_age'].iloc[indices])
                        # Shapiro test per group for multigroup test
                        group_data = G[G[col] == origin]['abs_PAD']
                        stat, p_value = stats.shapiro(group_data)
                        normality_p_values.append(p_value)
                    #-----------------------------------------------------------------------------
                    if len(G[col].unique()) > 1:
                        # Test for homogeneity of variances
                        stat, levene_p_value = stats.levene(
                            *[G[G[col] == group]['abs_PAD'] for group in G[col].unique()])
                        # 3. Choose the appropriate test based on the assumptions
                        if all(p > 0.05/num_tests for p in normality_p_values) and levene_p_value > 0.05:
                            stat_test = 'ANOVA'
                            model = ols('abs_PAD ~ C(' + col + ')', data=G).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            p_value_multigroup = anova_table["PR(>F)"][0]
                        else:
                            stat_test = 'Kruskal-Wallis'
                            stat, p_value_multigroup = stats.kruskal(*[G[G[col] == group]['abs_PAD'] for group in G[col].unique()])

                        groups_stat_PAD[healthy][col]['p_val'] = p_value_multigroup
                        groups_stat_PAD[healthy][col]['stat_test'] = stat_test
                    else:
                        groups_stat_PAD[healthy][col]['p_val'] = np.nan
                        groups_stat_PAD[healthy][col]['stat_test'] = np.nan

            df['MAE'+case] = mae['healthy_orx']
            df['MAE_cole'+case] = mae['healthy_cole']
            df['MAE_max_bin'+case] = max_bin_MAE['healthy_orx']
            df['MAE_max_bin_cole'+case] = max_bin_MAE['healthy_cole']
            df['MAE_group'+case] = str(mae_bins)
            df['mean_PAD_group'+case] = str(meanPAD_bins)
            df['MAE_db'+case] = str(groups_MAE['healthy_orx']['db_name'])
            df['MAE_db_cole'+case] = str(groups_MAE['healthy_cole']['db_name'])
            df['MAE_machine'+case] = str(groups_MAE['healthy_orx']['machine_model'])
            df['MAE_machine_cole'+case] = str(groups_MAE['healthy_cole']['machine_model'])
            df['MAE_manufacturer'+case] = str(groups_MAE['healthy_orx']['manufacturer'])
            df['MAE_manufacturer_cole'+case] = str(groups_MAE['healthy_cole']['manufacturer'])
            df['MAE_ethnicity'+case] = str(groups_MAE['healthy_orx']['ethnicity'])
            df['MAE_ethnicity_cole'+case] = str(groups_MAE['healthy_cole']['ethnicity'])
            df['MAE_education_years'+case] = str(groups_MAE['healthy_orx']['education_years'])
            df['MAE_education_years_cole'+case] = str(groups_MAE['healthy_cole']['education_years'])
            df['MAE_CN_type'+case] = str(groups_MAE['healthy_orx']['CN_type'])
            df['MAE_CN_type_cole'+case] = str(groups_MAE['healthy_cole']['CN_type'])
            df['multistat_db'+case] = str(groups_stat_PAD['healthy_orx']['db_name'])
            df['multistat_db_cole'+case] = str(groups_stat_PAD['healthy_cole']['db_name'])
            df['multistat_machine'+case] = str(groups_stat_PAD['healthy_orx']['machine_model'])
            df['multistat_machine_cole'+case] = str(groups_stat_PAD['healthy_cole']['machine_model'])
            df['multistat_manufacturer'+case] = str(groups_stat_PAD['healthy_orx']['manufacturer'])
            df['multistat_manufacturer_cole'+case] = str(groups_stat_PAD['healthy_cole']['manufacturer'])
            df['multistat_ethnicity'+case] = str(groups_stat_PAD['healthy_orx']['ethnicity'])
            df['multistat_ethnicity_cole'+case] = str(groups_stat_PAD['healthy_cole']['ethnicity'])
            df['multistat_education_years'+case] = str(groups_stat_PAD['healthy_orx']['education_years'])
            df['multistat_education_years_cole'+case] = str(groups_stat_PAD['healthy_cole']['education_years'])
            df['multistat_CN_type'+case] = str(groups_stat_PAD['healthy_orx']['CN_type'])
            df['multistat_CN_type_cole'+case] = str(groups_stat_PAD['healthy_cole']['CN_type'])
            df['Pearson_chronological_brain_age'+case] = str(pearson_age_bio)
            df['Pearson_age_PAD'+case] = str(pearson_age_PAD)
            df['R-squared'+case] = str(r_squared)
            df['MSE'+case] = str(mse)
        #-----------------------------------------------------------------------------------------------------------
        # STEP 2: Metrics if must be computed per bound

        if not save_metrics:
            self.metrics = df
        else:
            df.to_csv(self.directory_metrics + self.name + '.csv', index=False)

    def age_prediction_plots(self):
        """
        :return: This function is used to generate plots showcasing age prediction plots on test set for both healthy
                 population definition.
        """
        for healthy in ['healthy_orx', 'healthy_cole']:  # Consider both alternatives for CN definition
            X_test = self.X_test[self.X_test[healthy]]
            X_test = X_test.reset_index()
            # -------------------------------------------------------------------------------
            # STEP 0: Brain age and PAD plots over full test set
            plt.close()
            plt.rcParams.update(plt.rcParamsDefault)
            sns.set_style("darkgrid")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.scatterplot(data=X_test, x='age_at_scan', y='brain_age')
            plt.plot([X_test['age_at_scan'].min(), X_test['age_at_scan'].max()],
                     [X_test['age_at_scan'].min(), X_test['age_at_scan'].max()], 'r--')
            plt.title('Chronological Age vs Brain Age')
            plt.xlabel('Chronological Age')
            plt.ylabel('Brain Age')
            # -------------------------------------------------------------------------------
            # STEP 1: Create the scatter plot for 'chronological_age' vs 'PAD'
            plt.subplot(1, 2, 2)
            sns.scatterplot(data=X_test, x='age_at_scan', y='PAD')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Chronological Age vs PAD')
            plt.xlabel('Chronological Age')
            plt.ylabel('￼PAD')

            plt.tight_layout()
            plt.savefig(self.directory_plots + 'CN_test_' + healthy + '_' + self.name + '.png')

    def disease_prediction_performance(self, save_metrics=True,
                                       event_predictors=['brain_age', 'PAD']):
        """
        :param save_metrics: Boolean (True, False) to decide whether or not to save the computed metrics for age predition performance. Default: False.
        :param event_predictors: Indicators to use as disease predictors of the binary events of interest. Options: ['brain_age','PAD']. Default: ['brain_age','PAD'].
        :param events: Binary events to analyze. Options: ['dem_nodem', 'cn_nocn']. Default: ['dem_nodem', 'cn_nocn'].
        :return: In this function we assess the capacity of the provided indicators in predicting different neurodegenerative disease binary events.
        """
        if not save_metrics:
            df = pd.DataFrame([[self.name]], columns=['name'])
        else:
            df = self.metrics
        # -------------------------------------------------------------------------------
        # STEP 0: Determine events of interest (minority class must have more than event_limit_cases MRIs available)
        events = []
        for col in self.binary_columns_test:
            if min(self.binary_cases_test[col].values) > self.event_limit_cases:
                events.append(col)
        # -------------------------------------------------------------------------------
        # STEP 1: Compute performance metrics (binary classification): auroc, p-values
        for case in self.age_cases_to_analyze:
            for event in events:
                print('--- ', event, ' ---')
                df_case = self.X_test.dropna(subset=[event, 'brain_age', 'PAD', 'age_at_scan'])

                if case == '_bounded':
                    df_case = df_case[
                        (df_case['age_at_scan'] >= self.age_bound[0]) & (df_case['age_at_scan'] <= self.age_bound[1])]
                df_case = df_case.reset_index()

                event_counts = df_case[event].value_counts()

                min_case_event = event_counts.idxmin()
                df_case['binary_feat'] = df_case[event].apply(lambda x: 1 if x == min_case_event else 0)
                df['cardinality_auroc_' + event + case] = str(event_counts.to_dict())

                if (event_counts.min() >= 50)and(len(event_counts)>1):
                    const = 1
                    if min_case_event == 'CN':
                        const = -1
                    for predictor in event_predictors:
                        df['auroc_' + predictor + '_' + event + case] = roc_auc_score(df_case['binary_feat'],
                                                                               const * df_case[predictor])
                        #-------------------------------------------------------------------------------------------------------
                        # Statistical Test
                        group1 = df_case[df_case['binary_feat'] == 1][predictor]
                        group2 = df_case[df_case['binary_feat'] == 0][predictor]
                        shapiro_g1 = shapiro(group1)
                        shapiro_g2 = shapiro(group2)
                        levene_test = levene(group1, group2)

                        # Selection of test
                        if shapiro_g1.pvalue > 0.05/2 and shapiro_g2.pvalue > 0.05/2 and levene_test.pvalue > 0.05:
                            # If normality and equal variances are satisfied
                            test = ttest_ind(group1, group2)
                            stat_test = "Independent t-test"
                        elif shapiro_g1.pvalue > 0.05/2 and shapiro_g2.pvalue > 0.05/2:
                            # If normality is satisfied but variances are not equal
                            test = ttest_ind(group1, group2, equal_var=False)
                            stat_test = "Welch's t-test"
                        else:
                            # If normality is not satisfied
                            test = mannwhitneyu(group1, group2)
                            stat_test = "Mann-Whitney U-test"

                        df['pval_' + predictor + '_' + event + case] = test.pvalue
                        df['stat_test_' + predictor + '_' + event + case] = stat_test

                    if len(event_predictors) > 1:
                        df['auroc_' + event + case] = df[['auroc_brain_age_' + event + case, 'auroc_PAD_' + event + case]].max(axis=1)
                        df['pval_' + event + case] = df[['pval_brain_age_' + event + case, 'pval_PAD_' + event + case]].min(axis=1)
                    else:
                        df['auroc_' + event + case] = df['auroc_' + event_predictors[0] + '_' + event + case]
                        df['pval_' + event + case] = df['pval_' + event_predictors[0] + '_' + event + case]
                else:
                    for predictor in event_predictors:
                        df['auroc_' + predictor + '_' + event + case] = np.nan
                        df['pval_' + predictor + '_' + event + case] = np.nan
                        df['stat_test_' + predictor + '_' + event + case] = np.nan
                        df['auroc_' + event + case] = np.nan
                        df['pval_' + event + case] = np.nan

        if not save_metrics:
            self.metrics = df
        else:
            df.to_csv(self.directory_metrics + self.name + '.csv', index=False)

    def disease_prediction_plots(self, event_predictors=['brain_age', 'PAD']):
        """
        :param save_metrics: Boolean (True, False) to decide whether or not to save the computed metrics for age predition performance. Default: False.
        :param event_predictors: Indicators to use as disease predictors of the binary events of interest. Options: ['brain_age','PAD']. Default: ['brain_age','PAD'].
        :return:Plots to showcase brain age/PAD distribution on healthy individual's MRI vs MRIs of individuals with specific neurodegenerative conditions at t=0 (binary events).
        """
        # -------------------------------------------------------------------------------
        # STEP 1: Determine events of interest (minority class must have more than event_limit_cases MRIs available)
        '''
        events = []
        for col in self.binary_columns_test:
            if min(self.binary_cases_test[col].values) > self.event_limit_cases:
                events.append(col)'''
        # -------------------------------------------------------------------------------
        # STEP 2: Generate plots to show performance of indicator on healthy individuals' MRI vs various NDD groups.
        #print('---------------------------------------------------')
        #print('*',self.name)
        for predictor in event_predictors:
            plt.close()
            plt.rcParams.update(plt.rcParamsDefault)
            sns.set_style("darkgrid")
            plt.figure(figsize=(5 * len(self.events), 5))
            for idx, event in enumerate(self.events):
                plt.subplot(1, len(self.events), idx + 1)
                df_case = self.X_test.dropna(subset=[event, predictor, 'age_at_scan'])
                #print(predictor,event,df_case[[event, predictor, 'age_at_scan']].isna().sum())
                sns.histplot(data=df_case, x=predictor, hue=event, kde=True)
                # plt.yscale('log')
                plt.title('Distribution of ' + predictor + ' in ' + event)
                plt.xlabel(predictor)
            plt.tight_layout()
            plt.savefig(self.directory_plots + 'Disease_' + predictor + '_' + self.name + '.png')

    def pareto_preselection(self, features_pareto=['MAE', 'auroc_dem_nodem', 'auroc_cn_nocn'],
                            reduced_version=True):

        """
        :param features_pareto: Features to be considered to generate Pareto front.
        :param pareto_location: File location/name where dataframe with Pareto front is expected to be stored.
        :param reduced_version: Boolean (True, False) to decide whether or not to eliminate penalty based solutions with null penalty term (which should coincide with a regular linear regression).
        :param reduced_location: File location/name where dataframe with reduced representation of the full set of results is expected to be stored.
        :return: Pareto front as model selection tool.
        """

        # -------------------------------------------------------------------------------
        # STEP 0: Function to verify whether a model is dominated by the performance of a different candidate model.
        def dominates(row1, row2):
            return all(a <= b for a, b in zip(row1, row2)) and any(a < b for a, b in zip(row1, row2))

        # -------------------------------------------------------------------------------
        # STEP 1: Collect full set of data and generate its reduced representation.
        self.collector()
        self.results_reduced = self.results
        if self.save_analyzed_cases:
            self.results_reduced.to_csv(self.directory_pareto + 'analyzed_cases_' + str(features_pareto) + '.csv',
                                        index=False)

        # -------------------------------------------------------------------------------
        # STEP 2: Generate Pareto front maximizing all performance metrics.
        for col in features_pareto:  # just to make sure we want to minimize all indicators in cols
            if col.startswith('auroc'):
                self.results_reduced[col] *= -1

        pareto_front = []
        for _, row1 in self.results_reduced.iterrows():
            is_dominated = False
            for _, row2 in self.results_reduced.iterrows():
                if row1 is not row2 and dominates(row2[features_pareto], row1[features_pareto]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(row1['name'])

        for col in features_pareto:  # revert the previous
            if col.startswith('auroc'):
                self.results_reduced[col] *= -1

        self.results_reduced['pareto'] = self.results_reduced.apply(lambda row: row['name'] in pareto_front, axis=1)
        self.results_reduced = self.results_reduced[self.results_reduced['pareto']]
        self.results_reduced.reset_index(drop=True, inplace=True)
        self.results_reduced[['name','modelling_strategy', 'method', 'training_population', 'used_predictors',
                                       'used_predictors_set_size', 'sampling', 'MAE_training', 'MAE_group_training'] + features_pareto].to_csv(
            self.directory_pareto + 'pareto_' + str(features_pareto) + '.csv', index=False)

        if self.save_plots:
            for idx,name in enumerate(self.results_reduced['name']):
                df = pd.read_csv(self.directory_metrics_train + name + '.csv')
                pkl_name = df['save_model'].values[0]
                self.model = cl.LMB.load_model(self.directory_models + pkl_name + '.pkl')
                self.bias_type = df['age-bias'].values[0]
                self.name = name
                self.X_test['brain_age'] = self.model.predict(self.X_test, bias_type = self.bias_type,
                                                              y = self.X_test['age_at_scan'].values)
                self.X_test['PAD'] = self.X_test['brain_age'] - self.X_test['age_at_scan']
                self.age_prediction_plots()
                self.disease_prediction_plots()

    def collector(self):
        """
        :param all_results_location: Location/filename where dataframe with full set of metrics per model should be stored.
        :return: Unified dataframe with all quality metrics.
        """
        dataframes = []
        for filename in os.listdir(self.directory_metrics):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory_metrics, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        unique_df = pd.concat(dataframes, ignore_index=True)
        self.results = unique_df

    def update_history(self):
        """
        :return: Update associated .json file with already executed models train configuration/corrected models/tested models info
        """
        data = {'examined_combinations': self.examined_combinations, 'corrected_ids': self.corrected_ids, 'id': self.id,
                'tested': self.tested}
        with open(self.historic_filepath, 'w') as file:
            json.dump(data, file)

    def test_models_baseline_feats(self, features_pareto=['MAE', 'auroc_dem_nodem', 'auroc_cn_nocn']):
        """
        :param features_pareto: Test metric set of features to be considered when determining Pareto fronts (in this case just used for file naming).
        :return: Function to evaluate performance of the selected set of models in terms of age/disease prediction.
        """
        inverted_features = ['MOCA', 'MMSE']
        results_test_feats = []
        # -------------------------------------------------------------------------------
        # STEP 1: Compute performance metrics over models selected via Pareto (using selected test set subset including baseline metrics)
        for name in self.results_reduced['name']:
            pkl_name = self.cases_to_analyze[self.cases_to_analyze['name'] == name]['save_model'].values[0]
            self.model = cl.LMB.load_model(self.directory_models + pkl_name + '.pkl')
            self.bias_type = self.cases_to_analyze[self.cases_to_analyze['name'] == name]['age-bias'].values[0]
            self.name = name
            self.X_test_feats['brain_age'] = self.model.predict(self.X_test_feats, bias_type=self.bias_type,
                                                                y=self.X_test_feats['age_at_scan'].values)

            self.X_test_feats['PAD'] = self.X_test_feats['brain_age'] - self.X_test_feats['age_at_scan']
            corr_baseline_feats = [self.X_test_feats['brain_age'].corr(self.X_test_feats[col], method='pearson') for col
                                   in self.baseline_test_feats]
            name_baseline_feats = ['Pearson_corr_' + col for col in self.baseline_test_feats]
            self.metrics = pd.DataFrame([[self.name] + corr_baseline_feats],
                                        columns=['name'] + name_baseline_feats)
            self.prediction_performance()
            results_test_feats.append(self.metrics)
        # -------------------------------------------------------------------------------
        # STEP 2: Compute performance metrics of selected baseline metrics (using selected test set subset including baseline metrics)
        for name in self.baseline_test_feats:
            const2 = 1 if name not in inverted_features else -1
            self.name = name
            corr_baseline_feats = [self.X_test_feats[self.name].corr(self.X_test_feats[col], method='pearson') for col
                                   in self.baseline_test_feats]
            name_baseline_feats = ['Pearson_corr_' + col for col in self.baseline_test_feats]
            self.metrics = pd.DataFrame([[self.name] + corr_baseline_feats],
                                        columns=['name'] + name_baseline_feats)
            self.prediction_performance(predictor=self.name, const2=const2)
            results_test_feats.append(self.metrics)

        self.results_test_feats = pd.concat(results_test_feats, ignore_index=True)
        self.results_test_feats.to_csv(self.directory_pareto + 'baseline_feats_'+ str(self.baseline_test_feats) + '_pareto_' + str(features_pareto) + '.csv',
                                       index=False)

    def prediction_performance(self, predictor = 'brain_age', const2=1, stat_features=False):
        """
        :param predictor: Feature that we want to assess disease performance. Default: 'brain_age'.
        :param const2: Corrector to be multiplied to predictor. It is 1, if the larger the score is the more harmful, e.g., chronological age,
                 it is -1 if the opposite, e.g., MOCA score.
        :return: In this function, similarly to disease_prediction_performance(), we assess the capacity of the provided indicators
                 in predicting different neurodegenerative disease binary events and correlation metrics to age. Observe that we do
                 not compute all metrics of disease_prediction_performance() but just those of 'brain_age' or selected non-'brain_age'
                 features.
        """
        # -------------------------------------------------------------------------------
        # STEP 1: Compute performance metrics (binary classification): auroc, t-test, u-test
        for event in self.events_test_feats:
            df_case = self.X_test_feats.dropna(subset=[event, 'brain_age', 'PAD'])
            filtered_dict = {k: v for k, v in self.X_test_feats[event].value_counts(dropna=True).items() if
                             v is not None}
            min_case_event = min(filtered_dict, key=filtered_dict.get)
            df_case['binary_feat'] = df_case[event].apply(lambda x: 1 if x == min_case_event else 0)
            # -------------------------------------------------------------------------------
            # STEP 2: Modify predictor sign if needed (if CN is minority class, for instance)
            const = 1
            if min_case_event == 'CN':
                const = -1
            # -------------------------------------------------------------------------------
            # STEP 3: Compute binary disease prediction performance metrics
            roc_ = roc_auc_score(df_case['binary_feat'], const * const2 * df_case[predictor])
            if predictor == 'brain_age':
                roc_PAD = roc_auc_score(df_case['binary_feat'], const * const2 * df_case['PAD'])
                roc_ = max(roc_,roc_PAD)
            self.metrics['auroc_' + event] = roc_
            if stat_features:

                group1 = df_case[df_case['binary_feat'] == 1][predictor]
                group2 = df_case[df_case['binary_feat'] == 0][predictor]
                shapiro_g1 = shapiro(group1)
                shapiro_g2 = shapiro(group2)
                levene_test = levene(group1, group2)
                # Selection of test
                if shapiro_g1.pvalue > 0.05/3 and shapiro_g2.pvalue > 0.05/3 and levene_test.pvalue > 0.05/3:
                    # If normality and equal variances are satisfied
                    test = ttest_ind(group1, group2)
                    stat_test = "Independent t-test"
                elif shapiro_g1.pvalue > 0.05/3 and shapiro_g2.pvalue > 0.05/3:
                    # If normality is satisfied but variances are not equal
                    test = ttest_ind(group1, group2, equal_var=False)
                    stat_test = "Welch's t-test"
                else:
                    # If normality is not satisfied
                    test = mannwhitneyu(group1, group2)
                    stat_test = "Mann-Whitney U-test"
                p_value_t_ = test.pvalue

                if predictor == 'brain_age':

                    group1 = df_case[df_case['binary_feat'] == 1]['PAD']
                    group2 = df_case[df_case['binary_feat'] == 0]['PAD']
                    shapiro_g1 = shapiro(group1)
                    shapiro_g2 = shapiro(group2)
                    levene_test = levene(group1, group2)
                    # Selection of test
                    if shapiro_g1.pvalue > 0.05 / 3 and shapiro_g2.pvalue > 0.05 / 3 and levene_test.pvalue > 0.05 / 3:
                        # If normality and equal variances are satisfied
                        test = ttest_ind(group1, group2)
                        stat_test = "Independent t-test"
                    elif shapiro_g1.pvalue > 0.05 / 3 and shapiro_g2.pvalue > 0.05 / 3:
                        # If normality is satisfied but variances are not equal
                        test = ttest_ind(group1, group2, equal_var=False)
                        stat_test = "Welch's t-test"
                    else:
                        # If normality is not satisfied
                        test = mannwhitneyu(group1, group2)
                        stat_test = "Mann-Whitney U-test"
                    p_value_t_pad = test.pvalue
                    p_value_t_ = min(p_value_t_, p_value_t_pad)

                self.metrics['pval_' + event] = p_value_t_
