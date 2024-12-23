import numpy as np
import pandas as pd
import json
import warnings
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import contextlib
import sys

warnings.filterwarnings("ignore")

class dataset_selection():
    def __init__(self, data_location = 'fastsurfer_stats_legacy_vox1_harmonized_demographics_filtered_machine_type_death_record_last_follow_up.csv',
                 directory_data = 'data/', time_splits = [0,5,10], age_limits = None,
                 data_preselected = ['UKBB', 'ADNI', 'NACC'], data_test_only = ['ADNI','NACC'],
                 modalities_location = 't1_flair_dti_bold_boolean.csv', delta_exclusion_months = None):
        """
        :param data_location: Location of original database containing full info (demographic and features).
        :param modalities_location: Location of database containing modalities available per individual (this info is considered during the split).
        :param directory_data: Directory where we want to save all generated metadata. Default: Create subfolder named '/data'.
        :param time_splits: Time limits to be considered to generate CN subtypes. Default: [0,5,10]
        :param age_limits: Interval of age to be considered in the analysis. Default: None, i.e., we use all data points
        :return: Data split for training and test AVIV models
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.age_limits = age_limits
        self.data_preselected = data_preselected
        self.data_test_only = data_test_only
        self.data_train_also = [item for item in self.data_preselected if item not in self.data_test_only]
        self.full_data = pd.read_csv(data_location)
        self.full_data = self.full_data[self.full_data['db_name'].isin(self.data_preselected)]
        self.delta_exclusion_months = delta_exclusion_months
        modalities = pd.read_csv(modalities_location)
        self.full_data = pd.merge(self.full_data, modalities, on='unique_subj_id')


        if self.age_limits is not None:
            self.full_data = self.full_data[
                (self.full_data['age_at_scan'] >= self.age_limits[0]) & (self.full_data['age_at_scan'] <= self.age_limits[1])]
            self.full_data.reset_index(drop=True, inplace=True)

        self.directory_data = directory_data
        if not os.path.exists(self.directory_data):
            os.makedirs(self.directory_data, exist_ok=True)
        self.time_splits = time_splits
        self.preprocessing_type = 'age_' + str(self.age_limits) + '_time_splits_' + str(self.time_splits) + '_delta_exclusion_' + str(self.delta_exclusion_months)
        #---------------------------------------------------------------------------------------------------------------
        # STEP 1: Run data split generator with preselected settings and save printed messages in 'printout.txt'
        '''
        print('Initial input: Database with ' + str(len(self.full_data)) + ' MRIs coming from ' + str(
            len(self.full_data['unique_subj_id'].unique())) + ' individuals.')
        self.data_correction()
        self.data_split(delta_exclusion_months=self.delta_exclusion_months, stratified_by_CN=True)'''

        with open(self.directory_data + 'printout.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                print('Initial input: Database with ' + str(len(self.full_data)) + ' MRIs coming from ' + str(len(self.full_data['unique_subj_id'].unique())) + ' individuals.')
                self.data_correction()
                self.data_split(delta_exclusion_months = self.delta_exclusion_months, stratified_by_CN = True)

    def data_correction(self):
        """
        :return: Corrected version of the original database, eliminating samples for which gender is unknown,
                 elimininating missing cognitive test info and adding cognitive type info
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Eliminate cases where gender is unknown
        data = self.full_data.dropna(subset=['biological_sex'])
        data = data.reset_index(drop=True)
        #---------------------------------------------------------------------------------------------------------------
        # STEP 1: Correct cognitive test scores (nullify negative scores)
        data['MOCA'] = data['MOCA'].apply(lambda x: x if x >= 0 else float('nan'))
        data['MMSE'] = data['MMSE'].apply(lambda x: x if x >= 0 else float('nan'))
        data['CDR'] = data['CDR'].apply(lambda x: x if x >= 0 else float('nan'))
        #---------------------------------------------------------------------------------------------------------------
        #STEP 2: Divide analysis between UKB and additional databases as both have different preprocessing
        data_UK = data[data['db_name'] == 'UKBB']
        data = data[data['db_name'] != 'UKBB']
        #---------------------------------------------------------------------------------------------------------------
        # STEP 3: UKB analysis: Quantify year gap per disease and gather diseases by group ('relable'). Two columns added:
        #                      'Condition': Relabeling of condition (if not CN) considering 'relable'. 'CN_type': Specifying
        #                      'CN' and the maximum time_split to which we know the patient remains 'CN' post sample extraction
        data_UK['acquisition_date'] = pd.to_datetime(data_UK['acquisition_date'])
        data_UK['death_date'] = pd.to_datetime(data_UK['death_date'])
        data_UK['low_bound_date'] = pd.to_datetime('2016-05-31')
        data_UK['up_bound_date'] = pd.to_datetime('2024-05-31')

        #3.1:Convert dates of diagnosis/death to time to diagnosis/death (in years), taking into account date at scan
        cols = ['Dementia', 'VascularDementia', 'OtherDiseaseDementia', 'UnspecifiedDementia', 'Huntington',
                'PrimaryParkinson', 'SecondaryParkinson', 'Alzheimer', 'MultipleSclerosis', 'otherSpinalChord', 'death_date', 'low_bound_date','up_bound_date']
        for col in cols:
            data_UK[col] = pd.to_datetime(data_UK[col])
            data_UK[col] = (data_UK[col] - data_UK['acquisition_date']).dt.days
            data_UK[col] /= 365

        #3.2:Gather all dementia/parkinson-related diagnosis into a single group as specified in the dictionary 'relable'
        relable = {'Dementia': ['Dementia', 'VascularDementia', 'OtherDiseaseDementia', 'UnspecifiedDementia'],
                   'Parkinson': ['PrimaryParkinson', 'SecondaryParkinson']}
        data_UK['Dementia'] = np.nanmin(data_UK[relable['Dementia']], axis=1)
        data_UK['Parkinson'] = np.nanmin(data_UK[relable['Parkinson']], axis=1)
        cols = ['Dementia', 'Parkinson', 'Huntington', 'Alzheimer', 'MultipleSclerosis', 'otherSpinalChord', 'death_date', 'low_bound_date','up_bound_date']
        data_UK['Condition'] = np.nan
        data_UK['CN_type'] = np.nan
        data_UK['low_bound_date'] = data_UK['low_bound_date'].apply(lambda x: np.nan if x <= 0 else x)

        #3.3: Elimninate 'low_bound' (GP last update upper bound) if there is any diagnosis or death
        condition = data_UK[cols[:-2]].notna().any(axis=1)
        data_UK.loc[condition, 'low_bound_date'] = np.nan
        print('* ',data_UK['low_bound_date'].isna().sum(), ' cases out of ', len(data_UK), ' with extraction date later than 2016/05/31 (UKBB)')

        #3.4: Generate 'CN_type' and 'Condition' features
        for idx, row in data_UK.iterrows():
            cols2 = sorted(row[cols].dropna().items(), key=lambda x: (x[1], cols.index(x[0])))
            filtered_splits = [t for t in self.time_splits if t < cols2[0][1]]
            if len(filtered_splits) > 0:
                data_UK.at[idx, 'CN_type'] = 'CN_' + str(int(max(filtered_splits)))
            else:
                label = ''
                for disease in cols2:
                    if disease[1] <= 0:
                        label = label + disease[0] + '_'
                data_UK.at[idx, 'Condition'] = label[:-1]

        #3.5: Geneate 'healthy_orx' and 'healthy_cole' boolean columns. These are used to decide the cases to train and test
        #     the brain clock(s) following oRx or Cole criteria
        data_UK['healthy_orx'] = data_UK['CN_type'].notna()
        data_UK['healthy_cole'] = data_UK['cohort'] == 'CN'
        #---------------------------------------------------------------------------------------------------------------
        #Step 4: non-UKB analysis: Quantify year gap per disease/CN looking at the various MRI of an individual. Since
        #                         cognitive status may vary drastically, an individuals' CN status remain so as long as no
        #                         no-CN status is observed chronologically.
        data['Condition'] = data['cognitive_status'].apply(lambda x: x if x != 'CN' else np.nan)
        data['CN_type'] = np.nan
        for idx, row in data.iterrows():
            if row['cognitive_status'] == 'CN':
                same_patient = data[
                    (data['unique_subj_id'] == row['unique_subj_id']) & (data['age_at_scan'] > row['age_at_scan'])]
                same_patient = same_patient.sort_values(by='age_at_scan')
                gap = 0
                for idx2, row2 in same_patient.iterrows():
                    if row2['cognitive_status'] == 'CN':
                        gap = row2['age_at_scan'] - row['age_at_scan']
                    else:
                        break
                filtered_splits = [t for t in self.time_splits if t <= gap]
                data.at[idx, 'CN_type'] = 'CN_' + str(int(max(filtered_splits)))
        data['healthy_orx'] = data['CN_type'].notna()
        data['healthy_cole'] = data['healthy_orx']
        # ----------------------------------------------------------------------------------------
        #Step 5: Merge UKB and no UKB data and save it
        self.data = pd.concat([data, data_UK], ignore_index=True)
        self.data = self.data.reset_index(drop=True)
        self.data.to_csv(self.directory_data + 'full_data:' + self.preprocessing_type + '.csv', index=False)
        return

    def data_split(self, test_size = 0.20, delta_exclusion_months = None, stratified_by_CN = True):
        """
        :param test_size: Test dataset ratio to be considered when splitting the patients' population. Default: 0.20
        :param delta_exclusion_months: Delta MRI repetition interval to be considered to place in the test set. Default: None.
        :param stratified_by_CN: Whether or not CN_type must be considered in stratified splitting. If not, just consider 'healthy_orx'. Default: True.
        :return: Split of sef.data in training and test
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Eliminate from the set to split those samples corresponding to databases that must belong to either
        #        the training or test set mandatorily. Create 'target' feature, which contains the type of CN and gender
        #        as info to include in the stratified split
        db_ignore = self.data_test_only
        train_extra = False
        test_extra = False
        ratio = (1-test_size)/test_size

        full_data_ignore = {}
        if len(self.data_test_only) > 0:
            full_data_ignore['test'] = self.data[self.data['db_name'].isin(self.data_test_only)]
            test_extra = True
            print('* Step 0: Adding to test set ' + str(len(full_data_ignore['test'])) + ' MRIs from the ' + str(
                self.data_test_only) + ' preselected databases.')

        data_to_split = self.data[~self.data['db_name'].isin(db_ignore)]
        # ---------------------------------------------------------------------------------------------------------------
        # STEP 1: Take all data containing deltas of interest (regardless of cognitive status)
        if delta_exclusion_months is not None:

            def create_delta(df, delta):
                df['delta'] = False
                for i in range(len(df)):
                    current_age = df.loc[i, 'age_at_scan']
                    id = df.loc[i, 'unique_subj_id']
                    df2 = df[df['unique_subj_id'] == id]
                    condition = (df2['age_at_scan'] > current_age + delta[0]/12) & (df2['age_at_scan'] <= current_age + delta[1]/12)
                    cases = len(df2[condition])
                    if cases > 0:
                        df.loc[i, 'delta'] = True
                df = df[df['delta']]
                #-------------------------------------------------------------------------------------------------------
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
                fig.suptitle('Distributions of Variables in df')
                # Condition distribution
                sns.countplot(data=df, x='Condition', ax=axes[0, 0])
                axes[0, 0].set_title('Condition Distribution')
                # CN_type distribution
                sns.countplot(data=df, x='CN_type', ax=axes[0, 1])
                axes[0, 1].set_title('CN_type Distribution')
                # age_at_scan distribution
                sns.histplot(data=df, x='age_at_scan', kde=True, ax=axes[1, 0])
                axes[1, 0].set_title('Age at Scan Distribution')
                # biological_sex distribution
                sns.countplot(data=df, x='biological_sex', ax=axes[1, 1])
                axes[1, 1].set_title('Biological Sex Distribution')
                # db_name distribution
                sns.countplot(data=df, x='db_name', ax=axes[2, 0])
                axes[2, 0].set_title('Database Name Distribution')
                # Remove the empty subplot (bottom right)
                fig.delaxes(axes[2, 1])
                # Adjust layout
                plt.tight_layout()
                plt.subplots_adjust(top=0.95)
                plt.savefig(
                    self.directory_data + 'samples_with_copies_' + self.preprocessing_type + '.png')
                plt.show()
                print('* '+ str(len(df)) + ' images contain a MRI copy within the '+ str(delta_exclusion_months) +' months interval -CN (orx) ' + str(len(df[df['healthy_orx']])) + ', CN (Cole) ' + str(len(df[df['healthy_cole']])) + '.' )
                #-------------------------------------------------------------------------------------------------------
                return df['unique_subj_id'].unique().tolist()

            delta_ids = create_delta(data_to_split,delta_exclusion_months)
            additional_test = data_to_split[data_to_split['unique_subj_id'].isin(delta_ids)]
            data_to_split = data_to_split[~data_to_split['unique_subj_id'].isin(delta_ids)]
            additional_test = additional_test.drop(columns=['delta'])
            data_to_split = data_to_split.drop(columns=['delta'])
            if test_extra:
                full_data_ignore['test'] = pd.concat([full_data_ignore['test'], additional_test], ignore_index=True)
            else:
                full_data_ignore['test'] = additional_test
                test_extra = True
            print('* Step 1: Adding to test set ' + str(len(additional_test)) + ' MRIs -of ' + str(len(delta_ids)) +' individuals from ' + str(
                additional_test['db_name'].unique().tolist()) + ' databases with ' + str(
                delta_exclusion_months) + 'M deltas.')
        # ---------------------------------------------------------------------------------------------------------------
        # STEP 2: Extract those individuals that do not contain any CN case at any follow-up 'unique_subj_ids_with_all_nan'
        #        and place them in test set
        all_nan_subj_ids = data_to_split.groupby('unique_subj_id')['CN_type'].apply(lambda x: x.isna().all())
        unique_subj_ids_with_all_nan = all_nan_subj_ids[all_nan_subj_ids].index.tolist()
        additional_test = data_to_split[data_to_split['unique_subj_id'].isin(unique_subj_ids_with_all_nan)]

        if test_extra:
            full_data_ignore['test'] = pd.concat([full_data_ignore['test'], additional_test], ignore_index=True)
        else:
            full_data_ignore['test'] = additional_test
            test_extra = True
        print('* Step 2: Adding to test set ' + str(len(additional_test)) + ' MRIs from ' + str(
            additional_test['db_name'].unique().tolist()) + ' databases with no CN cases for a fixed individual.')
        data_to_split = data_to_split[~data_to_split['unique_subj_id'].isin(unique_subj_ids_with_all_nan)]
        #---------------------------------------------------------------------------------------------------------------
        # STEP 3: Function to extract and concatenate sorted unique CN_type values
        def concatenate_cn_types(sub_df):
            if stratified_by_CN:
                # Drop NaNs and extract the integer part, sort them, and join back to a string
                cn_types = sub_df['CN_type'].dropna().unique()
                cn_numbers = sorted([int(cn.split('_')[1]) for cn in cn_types])
                cn_tag = 'CN_'+str(cn_numbers[-1]) #'_'.join([f'CN_{num}' for num in cn_numbers])
            else:
                cn_tag = 'CN'
            return sub_df['biological_sex'].values[0] + '_' + sub_df['db_name'].values[0] + '_' + cn_tag #sub_df['biological_sex'].values[0] + '_' + sub_df['db_name'].values[0] + '_' + sub_df['modalities'].values[0] + '_' + cn_tag

        cn_concatenation = data_to_split.groupby('unique_subj_id').apply(concatenate_cn_types).reset_index()
        cn_concatenation.columns = ['unique_subj_id', 'CN_concatenation']
        data_to_split = data_to_split.merge(cn_concatenation, on='unique_subj_id', how='left')
        #---------------------------------------------------------------------------------------------------------------
        # STEP 4: Split the groups into train and test sets after
        CN_remaining = len(data_to_split['unique_subj_id'].unique())
        CN_train = 0

        if test_extra:
            df = full_data_ignore['test'][full_data_ignore['test']['healthy_orx']]
            df = df[df['db_name'].isin(self.data_train_also)]
            CN_test = len(df['unique_subj_id'].unique())
        else:
            CN_test = 0
        test_size = max(test_size/2, (CN_train + CN_remaining - ratio * CN_test)/((1 + ratio) * CN_remaining))

        print('* Step 3: Stratified split to be performed with a test_size of ' + str(
            test_size) + '. Remaining data corresponds to ' + str(data_to_split[
                                                                      'db_name'].value_counts()) + 'MRIs per database of which ' + str(data_to_split[data_to_split[
                                                                      'healthy_orx']][
                                                                      'db_name'].value_counts()) +' correspond to healthy individuals.')

        #---------------------------------------------------------------------------------------------------------------
        # STEP 5: The left individuals that belong to single-element classes are finally placed in the training set:
        class_counts = cn_concatenation['CN_concatenation'].value_counts()
        print('class_counts:', class_counts)

        filtered_df = cn_concatenation[cn_concatenation['CN_concatenation'].isin(class_counts[class_counts >= 2].index)]
        rejected = cn_concatenation[cn_concatenation['CN_concatenation'].isin(class_counts[class_counts < 2].index)]
        additional_train = data_to_split[data_to_split['unique_subj_id'].isin(rejected['unique_subj_id'].tolist())]

        if train_extra:
            full_data_ignore['train'] = pd.concat([full_data_ignore['train'], additional_train], ignore_index=True)
        else:
            full_data_ignore['train'] = additional_train
            train_extra = True
        print('* Step 4: Adding to train set ' + str(len(additional_train)) + ' MRIs from ' + str(len(rejected['unique_subj_id'].tolist())) + ' subjects belonging to classes with less than 2 cases.')
        #---------------------------------------------------------------------------------------------------------------
        # STEP 6: Perform split:
        train_groups, test_groups = train_test_split(filtered_df['unique_subj_id'], test_size = test_size,
                                                     stratify=filtered_df['CN_concatenation'], random_state=42)
        # Filter the original dataframe to get the train and test sets
        self.train_data = data_to_split[data_to_split['unique_subj_id'].isin(train_groups)]
        self.test_data = data_to_split[data_to_split['unique_subj_id'].isin(test_groups)]

        if test_extra:
            self.test_data = pd.concat([self.test_data, full_data_ignore['test']], ignore_index=True)

        common_ids = pd.Series(list(set(self.train_data['unique_subj_id']).intersection(set(self.test_data['unique_subj_id']))))
        print("-" + str(len(self.train_data['unique_subj_id'].unique())) + " patients in training set and " + str(len(self.test_data['unique_subj_id'].unique())) + " patients in test set-")
        print("-Unique 'unique_subj_id' values common between train_data and test_data out of:", common_ids + '-')
        #---------------------------------------------------------------------------------------------------------------
        # STEP 7: Create binary columns of interest (CN vs diseases). Train and test sets are treated separately as
        #          we may have different conditions in both cases
        self.binary_columns_train = []
        self.binary_cases_train = {}

        for CN_population_binary in ['orx','cole']:
            for condition in self.train_data['Condition'].unique():
                if str(condition) != 'nan':
                    #print(condition)
                    self.train_data['CN_'+condition+'_'+CN_population_binary] = np.where(self.train_data['Condition'] == condition, condition,
                                                  np.where(self.train_data['healthy_' + CN_population_binary], 'CN', np.nan))
                    self.binary_columns_train.append('CN_'+condition+'_'+CN_population_binary)
                    self.binary_cases_train['CN_'+condition+'_'+CN_population_binary] = self.train_data['CN_'+condition+'_'+CN_population_binary].value_counts(dropna=True)

            self.train_data['CN_noCN'+'_'+CN_population_binary] = np.where(self.train_data['healthy_' + CN_population_binary], 'CN', 'noCN')
            self.binary_columns_train.append('CN_noCN'+'_'+CN_population_binary)
            self.binary_cases_train['CN_noCN'+'_'+CN_population_binary] = self.train_data['CN_noCN'+'_'+CN_population_binary].value_counts(dropna=True)

        self.binary_columns_test = []
        self.binary_cases_test = {}
        for CN_population_binary in ['orx', 'cole']:
            for condition in self.test_data['Condition'].unique():
                if str(condition) != 'nan':#condition is not np.nan:
                    self.test_data['CN_'+condition+'_'+CN_population_binary] = np.where(self.test_data['Condition'] == condition, condition,
                                                  np.where(self.test_data['healthy_' + CN_population_binary], 'CN', np.nan))
                    self.binary_columns_test.append('CN_'+condition+'_'+CN_population_binary)
                    self.binary_cases_test['CN_'+condition+'_'+CN_population_binary] = self.test_data['CN_'+condition+'_'+CN_population_binary].value_counts(dropna=True)

            self.test_data['CN_noCN'+'_'+CN_population_binary] = np.where(self.test_data['healthy_' + CN_population_binary], 'CN', 'noCN')
            self.binary_columns_test.append('CN_noCN'+'_'+CN_population_binary)
            self.binary_cases_test['CN_noCN'+'_'+CN_population_binary] = self.test_data['CN_noCN'+'_'+CN_population_binary].value_counts(dropna=True)
        #---------------------------------------------------------------------------------------------------------------
        #  STEP 8: Adding some extra columns of interest and storing all columns considered to extract final test set.
        self.full_set_of_predictors = ['CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior',
                                       'CC_Anterior', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'CSF',
                                       'WM-hypointensities']

        ctx_features = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal',
                        'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
                        'middletemporal', 'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis',
                        'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
                        'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal', 'transversetemporal', 'insula']
        LR_features = ['Cerebral-White-Matter', 'Lateral-Ventricle', 'Inf-Lat-Vent', 'Cerebellum-White-Matter', 'Cerebellum-Cortex',
                       'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens-area', 'VentralDC', 'choroid-plexus']

        self.train_data['PTGENDER'] = self.train_data['biological_sex'].apply(lambda x: 1 if x == 'Male' else 0)
        self.test_data['PTGENDER'] = self.test_data['biological_sex'].apply(lambda x: 1 if x == 'Male' else 0)

        for col in ctx_features:
            self.train_data[col] = self.train_data['ctx-rh-' + col] + self.train_data['ctx-lh-' + col]
            self.test_data[col] = self.test_data['ctx-rh-' + col] + self.test_data['ctx-lh-' + col]
            self.full_set_of_predictors+=[col, 'ctx-rh-' + col, 'ctx-lh-' + col]

        for col in LR_features:
            self.train_data[col] = self.train_data['Right-' + col] + self.train_data['Left-' + col]
            self.test_data[col] = self.test_data['Right-' + col] + self.test_data['Left-' + col]
            self.full_set_of_predictors += [col, 'Right-' + col, 'Left-' + col]

        self.train_data.to_csv(self.directory_data + 'train_data:' + self.preprocessing_type + '.csv', index=False)
        self.test_data.to_csv(self.directory_data + 'test_data:' + self.preprocessing_type + '.csv', index=False)

        with open(self.directory_data + 'binary_cases' + self.preprocessing_type + '.pkl', 'wb') as file:
            pickle.dump((self.full_set_of_predictors, self.binary_cases_train, self.binary_cases_test, self.binary_columns_train, self.binary_columns_test), file)
        #---------------------------------------------------------------------------------------------------------------
        #  STEP 9: Plot dataset distribution in terms of all features of interest.
        self.plot_database()


    def plot_database(self):
        """
        :return: Plot distribution of all features of interest for both training/test set and print ratios.
        """
        directory_data_plot = self.directory_data + 'plots/'
        if not os.path.exists(directory_data_plot):
            os.makedirs(directory_data_plot, exist_ok=True)

        for test_type in ['full', 'split']:
            print('################ TEST TYPE:' + test_type + ' #################')
            for CN_type in ['orx', 'cole']:
                print('---------------- ' + CN_type + ' -----------------')
                train_data = self.train_data[self.train_data['healthy_' + CN_type]]
                test_data = self.test_data[self.test_data['healthy_' + CN_type]]

                if test_type == 'split':
                    test_data = test_data[test_data['db_name'].isin(self.data_train_also)]
                #---------------------------------------------------------------------------------------------------------------
                # STEP 1: Age distribution per train and test split
                plt.figure(figsize=(10, 6))
                sns.histplot(train_data['age_at_scan'], color='red', label='Train Data', kde=True, bins=10)
                sns.histplot(test_data['age_at_scan'], color='blue', label='Test Data', kde=True, bins=10)
                plt.xlabel('Age at Scan')
                plt.ylabel('Frequency')
                plt.title('Distribution of Age at Scan for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':age_at_scan_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                #---------------------------------------------------------------------------------------------------------------
                # STEP 2: Age distribution per train and test split
                plt.figure(figsize=(10, 3))
                sns.histplot(train_data['biological_sex'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['biological_sex'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Biological Sex')
                plt.ylabel('Frequency')
                plt.title('Distribution of Biological Sex for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':biological_sex_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                #---------------------------------------------------------------------------------------------------------------
                # STEP 3: Age distribution per train and test split
                plt.figure(figsize=(10, 3))
                sns.histplot(train_data['db_name'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['db_name'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Database')
                plt.ylabel('Frequency')
                plt.title('Distribution of Database for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':database_origin_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                print('-----------------------------------------------------')
                value_counts = train_data['db_name'].dropna().value_counts()
                total_count = len(train_data['db_name'].dropna())
                print('training_ratio (db_name):',value_counts / total_count)
                value_counts = test_data['db_name'].dropna().value_counts()
                total_count = len(test_data['db_name'].dropna())
                print('test_ratio (db_name):',value_counts / total_count)
                #---------------------------------------------------------------------------------------------------------------
                # STEP 3: Age distribution per train and test split
                plt.figure(figsize=(10, 4))
                sns.histplot(train_data['CN_type'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['CN_type'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Class of CN')
                plt.ylabel('Frequency')
                plt.title('Distribution of Class of CN for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':CN_class_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()

                print('-----------------------------------------------------')
                value_counts = train_data['CN_type'].dropna().value_counts()
                total_count = len(train_data['CN_type'].dropna())
                print('training_ratio (CN_type):',value_counts / total_count)
                value_counts = test_data['CN_type'].dropna().value_counts()
                total_count = len(test_data['CN_type'].dropna())
                print('test_ratio (CN_type):',value_counts / total_count)
                #---------------------------------------------------------------------------------------------------------------
                # STEP 4: Gender distribution per train and test split
                plt.figure(figsize=(10, 4))
                sns.histplot(train_data['biological_sex'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['biological_sex'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Gender Per Patient')
                plt.ylabel('Frequency')
                plt.title('Distribution of Gender for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':gender_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                print('-----------------------------------------------------')
                value_counts = train_data['biological_sex'].dropna().value_counts()
                total_count = len(train_data['biological_sex'].dropna())
                print('training_ratio (gender):',value_counts / total_count)
                value_counts = test_data['biological_sex'].dropna().value_counts()
                total_count = len(test_data['biological_sex'].dropna())
                print('test_ratio (gender):',value_counts / total_count)
                #---------------------------------------------------------------------------------------------------------------
                # STEP 5: Modalities per train and test split
                plt.figure(figsize=(10, 4))
                sns.histplot(train_data['modalities'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['modalities'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Modalities Per Patient')
                plt.ylabel('Frequency')
                plt.title('Distribution of Modalities for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':modalities_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                print('-----------------------------------------------------')
                value_counts = train_data['modalities'].dropna().value_counts()
                total_count = len(train_data['modalities'].dropna())
                print('training_ratio (modealities):',value_counts / total_count)
                value_counts = test_data['modalities'].dropna().value_counts()
                total_count = len(test_data['modalities'].dropna())
                print('test_ratio (modalities):',value_counts / total_count)
                #---------------------------------------------------------------------------------------------------------------
                # STEP 6: Age distribution per train and test split
                plt.figure(figsize=(10, 4))
                sns.histplot(train_data['manufacturer'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['manufacturer'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Modalities Per Patient')
                plt.ylabel('Frequency')
                plt.title('Distribution of Manufacturer for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':manufacturer_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                print('-----------------------------------------------------')
                value_counts = train_data['manufacturer'].dropna().value_counts()
                total_count = len(train_data['manufacturer'].dropna())
                print('training_ratio (manufacturer):',value_counts / total_count)
                value_counts = test_data['manufacturer'].dropna().value_counts()
                total_count = len(test_data['manufacturer'].dropna())
                print('test_ratio (manufacturer):',value_counts / total_count)
                #---------------------------------------------------------------------------------------------------------------
                # STEP 7: Age distribution per train and test split
                plt.figure(figsize=(15, 4))
                sns.histplot(train_data['machine_model'], color='red', label='Train Data', stat="count", discrete=True)
                sns.histplot(test_data['machine_model'], color='blue', label='Test Data', stat="count", discrete=True)
                plt.xlabel('Machine Model Per Patient')
                plt.ylabel('Frequency')
                plt.title('Distribution of Machine Model for Train and Test Data')
                plt.legend()
                plt.savefig(directory_data_plot + test_type + ':machine_model_distribution_' + CN_type + '_' + self.preprocessing_type + '.png')
                plt.show()
                print('-----------------------------------------------------')
                value_counts = train_data['machine_model'].dropna().value_counts()
                total_count = len(train_data['machine_model'].dropna())
                print('training_ratio (machine_model):',value_counts / total_count)
                value_counts = test_data['machine_model'].dropna().value_counts()
                total_count = len(test_data['machine_model'].dropna())
                print('test_ratio (machine_model):',value_counts / total_count)



