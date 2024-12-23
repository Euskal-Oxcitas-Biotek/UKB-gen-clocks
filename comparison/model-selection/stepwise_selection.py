import numpy as np
import pandas as pd
import os
import pareto_selection as ps
import warnings
import itertools
warnings.filterwarnings("ignore")

class model_selection():
    def __init__(self, database_step = [['UKBB'],['ADNI','NACC']], gender = 'M', origin = ['SV', 'MC'],
                 features_pareto_step = 2*[['MAE_orx_bounded','max_MAE_bin_orx_bounded','auroc_bounded_CNvsNoCN']],
                 non_pareto_features=['Model_type', 'Model_name', 'modelling_strategy', 'Description', 'Age_bias_correction',
                                      'features_type', 'Training_healthy_group', 'Oversampling', 'Training_sex','used_predictors','used_predictors_set_size'],
                 directory_results = 'pareto_fronts/test_1/', age_bias_exclusion = False, generalization_cases = []):
        """
        :param database_step: List of databases sorted in the order we want to incorporate them in the analysis. Default: [['UKBB'],['ADNI','NACC']].
        :param gender: Gender we want to analyze. Default: Male ('M').
        :param origin: Specify origin of the database metrics to analyze. At the moment 'SV' and 'MC' available.
        :param features_pareto_step: List (of the same size as database_step) of the metrics we want to use for the Pareto selection.
               Default: 2*[['MAE_orx_bounded','max_MAE_bin_orx_bounded','auroc_bounded_CNvsNoCN']].
        :param directory_results: Directory where we want to save Pareto fronts. Default: Create subfolder named 'pareto_fronts/'.
        :param age_bias_exclusion: Boolean to determine if to exclude age-bias corrected models. Default: False.
        :param non_pareto_features: Non-numeric features/columns from the dataframe at data_location used to identify the models in the file with pareto front. Default:
               ['name', 'modelling_strategy', 'method', 'age-bias', 'features_type', 'training_population', 'sampling', 'gender']
        :param generalization_cases: List with thet type of features that we want to consider when assessing the generabizability of the model (mean/max of the maximum gap of MAE per type of feature).
               Full list of options: ['database', 'manufacturer','machine','ethnicity','CN_type','education_years']. If not needed, please set it as [].
        :return: Stepwise Pareto front
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.database_step = database_step
        self.features_pareto_step = features_pareto_step
        self.gender = gender
        self.origin = origin
        self.age_bias_exclusion = age_bias_exclusion
        self.non_pareto_features = non_pareto_features
        self.directory_results = directory_results
        self.generalization_cases = generalization_cases

    def stepwise_pareto(self):
        """
        :return: Stepwise Pareto, eliminate features sequentially via Pareto
        """
        self.pareto_front = None
        for i in range(len(self.database_step)):
            database = self.database_step[i]
            features_pareto = self.features_pareto_step[i]
            self.data_prep(database=database,features_pareto=features_pareto)
            print('cases per Model_type:', self.full_data['Model_type'].value_counts())
            if self.pareto_front is not None: #just keep the models available in the previous pareto
                self.full_data = self.full_data[self.full_data['Model_name'].isin(self.pareto_front['Model_name'])]
            dp = ps.model_selection(data_location = self.full_data, features_pareto = features_pareto,
                                    non_pareto_features = self.non_pareto_features, directory_results = self.directory_results, save_csv = True,
                                    csv_name = 'pareto_#'+str(i+1)+'_' + 'data_'+str(database)+'_age_biased_exclusion_'+str(self.age_bias_exclusion)+'.csv')
            dp.fit()
            self.pareto_front = dp.pareto_front


    def data_prep(self, database, features_pareto):
        """
        :param database: List of fixed-step databases to analyze.
        :return: Dataframe to extract Pareto front.
        """

        def max_pairwise_difference(row, columns, mode):
            values = row[columns].values
            pairwise_diffs = [np.abs(val1 - val2) for val1, val2 in itertools.combinations(values, 2)]
            if mode == 'max':
                indicator = max(pairwise_diffs)
            elif mode == 'median':
                indicator = np.median(pairwise_diffs)
            return indicator

        full_dataframes = []
        self.feature_cases_extractor(features_pareto) #determine healthy/bounded cases to be considered

        for origin in self.origin:
            dataframes = []
            for db in database:
                data = pd.read_csv('input_files/metrics_g'+self.gender+'_d'+db+'_c'+origin+'.csv')
                data = data.reindex(columns=data.columns.union(self.non_pareto_features), fill_value=np.nan)
                # ------------------------------------------------------------------------------------------------------
                # STEP 1: Eliminate corrected models if requested / and models trained on opposite gender
                if self.age_bias_exclusion:
                    data = data[data['Age_bias_correction'] == 'none']
                if self.gender == 'M':
                    data = data[data['Training_sex']!='female']
                elif self.gender == 'F':
                    data = data[data['Training_sex']!='male']
                # ------------------------------------------------------------------------------------------------------
                # STEP 2: Compute max columns (MAE,meanPAD)
                for healthy in self.healthy_cases:
                    for bounded in self.bounded_cases:
                        if bounded == 'unbounded':
                            MAE_cols =['MAE_' + healthy + '_age.[55,60]', 'MAE_' + healthy + '_age.[60,65]',
                                       'MAE_' + healthy + '_age.[65,70]', 'MAE_' + healthy + '_age.[70,75]',
                                       'MAE_' + healthy + '_age.[75,80]', 'MAE_' + healthy + '_age.[80,85]']
                            meanPAD_cols = ['meanPAD_'+healthy+'_age.[55,60]', 'meanPAD_'+healthy+'_age.[60,65]', 'meanPAD_'+healthy+'_age.[65,70]',
                                            'meanPAD_'+healthy+'_age.[70,75]', 'meanPAD_'+healthy+'_age.[75,80]', 'meanPAD_'+healthy+'_age.[80,85]']
                        else:
                            MAE_cols = ['MAE_' + healthy + '_age.[55,60]', 'MAE_' + healthy + '_age.[60,65]',
                                        'MAE_' + healthy + '_age.[65,70]', 'MAE_' + healthy + '_age.[70,75]',
                                        'MAE_' + healthy + '_age.[75,80]']
                            meanPAD_cols = ['meanPAD_' + healthy + '_age.[55,60]','meanPAD_' + healthy + '_age.[60,65]', 'meanPAD_' + healthy + '_age.[65,70]',
                                            'meanPAD_' + healthy + '_age.[70,75]', 'meanPAD_' + healthy + '_age.[75,80]']

                        data['max_MAE_bin_'+healthy+'_'+bounded] = data[MAE_cols].max(axis=1)
                        data['max_abs_meanPAD_'+healthy+'_'+ bounded] = data[meanPAD_cols].abs().max(axis=1)

                        data['abs(corr[PAD,CA]_' + healthy + '_' + bounded + ')'] = data[
                            'corr[PAD,CA]_' + healthy + '_' + bounded].abs()
                # ------------------------------------------------------------------------------------------------------
                # STEP 3: Construction of generabizability columns
                if len(self.generalization_cases) > 0:
                    sets_of_MAEs = {'database': ['db.ADNI', 'db.NACC', 'db.UKBB'],
                                    'manufacturer': ['manufacturer.GE', 'manufacturer.Philips', 'manufacturer.Siemens'],
                                    'machine': ['machine.Achieva', 'machine.Allegra', 'machine.Avanto',
                                                'machine.Biograph.mMR', 'machine.DISCOVERY.MR750', 'machine.Espree',
                                                'machine.GEMINI', 'machine.Gyroscan.NT',
                                                'machine.Ingenia', 'machine.Ingenuity', 'machine.Intera',
                                                'machine.NUMARIS/4', 'machine.Prisma', 'machine.SIGNA', 'machine.Skyra',
                                                'machine.Sonata', 'machine.Symphony',
                                                'machine.TrioTim', 'machine.Verio'],
                                    'ethnicity': ['ethnicity.Asian', 'ethnicity.Black', 'ethnicity.Caribbean',
                                                  'ethnicity.Chinese', 'ethnicity.Mixed', 'ethnicity.NativeAmerican',
                                                  'ethnicity.Other', 'ethnicity.White'],
                                    'CN_type': ['CN.0', 'CN.5', 'CN.10'],
                                    'education_years': ['education.[0,7]', 'education.[8,15]', 'education.[16,30]']} # Columns to look after per generalization type

                    generalization_entries = [entry[len('generalization_'):] for entry in features_pareto if entry.startswith('generalization_')]
                    self.final_generalization_cases = {}
                    for indicator in generalization_entries:
                        self.final_generalization_cases[indicator] = []
                        for healthy in self.healthy_cases:
                            for age_bounded in self.bounded_cases:
                                for key in self.generalization_cases:
                                    columns = ['MAE_'+ healthy + '_' + age_bounded + '_' + col for col in sets_of_MAEs[key] if 'MAE_'+ healthy + '_' + age_bounded + '_' + col in data.columns]
                                    subset_columns = [col for col in columns if data[col].notna().sum() == len(data)] #keep columns that are not nan for all models
                                    if len(subset_columns) > 1:
                                        data[key+ '_'+healthy+'_'+bounded+'_'+indicator] = data.apply(lambda row: max_pairwise_difference(row, subset_columns, indicator),
                                                                 axis=1)
                                        self.final_generalization_cases[indicator].append(key+ '_'+healthy+'_'+bounded+'_'+indicator)
                        data['generalization_'+indicator] = data[self.final_generalization_cases[indicator]].max(axis=1)
                        data = data.drop(columns=self.final_generalization_cases[indicator])
                # ------------------------------------------------------------------------------------------------------
                # STEP 4: Compute max auroc per disease (maximum between PAD and BA)
                for condition in ['CNvsNoCN', 'CNvsCondition.MultipleSclerosis', 'CNvsCondition.Dementia',
                                  'CNvsCondition.MCI', 'CNvsCondition.SMC', 'CNvsCondition.Parkinson',
                                  'CNvsCondition.OtherSpinalChord',
                                  'CNvsCondition.Alzheimer', 'CNvsCondition.Dementia.Alzheimer',
                                  'CNvsCondition.OtherSpinalChord.Dementia',
                                  'CNvsCondition.OtherSpinalChord.MultipleSclerosis',
                                  'CNvsCondition.OtherSpinalChord.Parkinson', 'CNvsCondition.Alzheimer.Dementia']:
                    for bound_case in ['bounded', 'unbounded']:
                        data['auroc_' + bound_case + '_' + condition] = data[
                            ['aurocPAD_' + bound_case + '_' + condition,
                             'aurocBA_' + bound_case + '_' + condition]].max(axis=1)
                dataframes.append(data[self.non_pareto_features + features_pareto])
            combined_data = pd.concat(dataframes, ignore_index=True)
            # ------------------------------------------------------------------------------------------------------
            # STEP 4: Merge results per dataframe keeping the worst case scenario across DBs
            results = []
            for model_name in combined_data['Model_name'].unique():
                model_data = combined_data[combined_data['Model_name'] == model_name]
                max_values = model_data[features_pareto].max() #worst case
                min_values = model_data.filter(regex=r'^(auroc|corr\[BA,CA\]|Rsquared)').min()#for these columns the worst case is the min
                result = {
                    'Model_name': model_name,
                    **max_values.to_dict(),  # Convert to dictionary
                    **min_values.to_dict()  # Convert to dictionary
                }
                # 4.1. Append the non_pareto_features values (same for a fixed Model_name)
                for feature in self.non_pareto_features:
                    result[feature] = model_data[feature].iloc[
                        0]  # Assuming values are the same for each Model_name
                # 4.2. Add result to the results list
                results.append(result)
            # 4.3. Create the DataFrame from the results list
            df = pd.DataFrame(results)
            full_dataframes.append(df)
        self.full_data = pd.concat(full_dataframes, ignore_index=True)

    def feature_cases_extractor(self,features_pareto):
        """
        :param features_pareto: Corresponding features_pareto at a given step.
        :return: Function to determine which healthy and bounded cases must be considered when generating the new set of features
        """
        self.healthy_cases = []
        self.bounded_cases = []

        # Check for 'orx' and 'cole' in elements of features_pareto
        if any('orx' in feature for feature in features_pareto):
            self.healthy_cases.append('orx')
        if any('cole' in feature for feature in features_pareto):
            self.healthy_cases.append('cole')

        # Check for '_bounded' and '_unbounded' in elements of features_pareto
        if any('_bounded' in feature for feature in features_pareto):
            self.bounded_cases.append('bounded')
        if any('_unbounded' in feature for feature in features_pareto):
            self.bounded_cases.append('unbounded')

