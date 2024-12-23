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
import ast

warnings.filterwarnings("ignore")

class plots_UKBB():
    def __init__(self, data_location = "pareto/Male_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN'].csv",
                 directory_results = 'post_analysis/Male/', type_of_approach = 'uncorrected', healthy_definition = 'orx',
                 keep_originals = False, binary_case = "CN_noCN", age_bounded = False):
        """
        :param data_location: Location of original database containing full info (demographic and features).
        :param directory_data: Directory where we want to save all generated metadata. Default: Create subfolder named '/data'.
        :return: Data split for training and test AVIV models
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.type_of_approach = type_of_approach
        self.full_data = pd.read_csv(data_location)
        self.healthy_definition = healthy_definition
        self.keep_originals = keep_originals
        self.binary_case = binary_case
        self.directory_results = directory_results
        self.age_bounded = age_bounded

        def modify_name(name):
            if name.endswith('_lange'):
                return name[:-6]
            elif name.endswith('_cole'):
                return name[:-5]
            elif name.endswith('_zhang'):
                return name[:-5]
            else:
                return name

        def age_bias(name):
            if name.endswith('_lange'):
                return 'lange'
            elif name.endswith('_cole'):
                return 'cole'
            elif name.endswith('_zhang'):
                return 'zhang'
            else:
                return np.nan

        self.full_data['case'] = self.full_data['name'].apply(modify_name)
        self.full_data['age-bias'] = self.full_data['name'].apply(age_bias)
        self.full_data['age-bias'].fillna('uncorrected', inplace=True)
        self.full_data['sampling'].fillna('None', inplace=True)
        #---------------------------------------------------------------------------------------------------------------
        feats = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal',
                 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
                 'middletemporal', 'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis',
                 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
                 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal', 'transversetemporal', 'insula',
                 'Cerebral-White-Matter', 'Lateral-Ventricle', 'Inf-Lat-Vent', 'Cerebellum-White-Matter', 'Cerebellum-Cortex',
                 'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens-area', 'VentralDC', 'choroid-plexus']

        def determine_feature_type(used_predictors_str, feats):
            used_predictors = ast.literal_eval(used_predictors_str)  # Convert string to list
            intersection = set(used_predictors).intersection(feats)
            return 'all' if intersection else 'merged'

        # Apply the function to create the 'features_type' column
        self.full_data['features_type'] = self.full_data['used_predictors'].apply(determine_feature_type, feats=feats)
        self.full_data['methodology'] = self.full_data['method'] + '_' + self.full_data['training_population'] + '_' + self.full_data['gender'] + '_' + self.full_data[
            'sampling'] + '_' + self.full_data['features_type']

        #Convert strings back to dictionary columns
        self.full_data['MAE_group'] = self.full_data['MAE_group'].apply(ast.literal_eval)
        self.full_data['MAE_db'] = self.full_data['MAE_db'].apply(ast.literal_eval)
        self.full_data['MAE_db_cole'] = self.full_data['MAE_db_cole'].apply(ast.literal_eval)
        self.full_data['MAE_machine'] = self.full_data['MAE_machine'].apply(ast.literal_eval)
        self.full_data['MAE_machine_cole'] = self.full_data['MAE_machine_cole'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer'] = self.full_data['MAE_manufacturer'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer_cole'] = self.full_data['MAE_manufacturer_cole'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type'] = self.full_data['MAE_CN_type'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type_cole'] = self.full_data['MAE_CN_type_cole'].apply(ast.literal_eval)
        self.full_data['Pearson_chronological_brain_age'] = self.full_data['Pearson_chronological_brain_age'].apply(ast.literal_eval)
        self.full_data['Pearson_age_PAD'] = self.full_data['Pearson_age_PAD'].apply(ast.literal_eval)
        self.full_data['Pearson_chronological_brain_age_bounded'] = self.full_data['Pearson_chronological_brain_age_bounded'].apply(ast.literal_eval)
        self.full_data['Pearson_age_PAD_bounded'] = self.full_data['Pearson_age_PAD_bounded'].apply(ast.literal_eval)

        def flatten_dict_double(d):
            return {f'{outer_key}_{inner_key}': value
                    for outer_key, inner_dict in d.items()
                    for inner_key, value in inner_dict.items()}

        def create_columns(d, prefix):
            return {f'{prefix}_{key}': value for key, value in d.items()}

        # Apply the function to the 'MAE_group' column and expand the result into new columns
        flattened_df = self.full_data['MAE_group'].apply(flatten_dict_double).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_group']).join(flattened_df)
        new_columns_df = self.full_data['MAE_db'].apply(lambda x: create_columns(x, 'healthy_orx')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_db_cole'].apply(lambda x: create_columns(x, 'healthy_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db_cole']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_age_PAD'].apply(lambda x: create_columns(x, 'Pearson_age_PAD')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_age_PAD']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_chronological_brain_age'].apply(lambda x: create_columns(x, 'Pearson_chronological_brain_age')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_chronological_brain_age']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_age_PAD_bounded'].apply(lambda x: create_columns(x, 'Pearson_age_PAD_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_age_PAD_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_chronological_brain_age_bounded'].apply(lambda x: create_columns(x, 'Pearson_chronological_brain_age_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_chronological_brain_age_bounded']).join(new_columns_df)
        #---------------------------------------------------------------------------------------------------------------
        # STEP 1: Run data split generator with preselected settings and save printed messages in 'printout.txt'
        if not os.path.exists(self.directory_results):
            os.makedirs(self.directory_results, exist_ok=True)

        self.directory_results_global = self.directory_results + 'global/'
        if not os.path.exists(self.directory_results_global):
            os.makedirs(self.directory_results_global, exist_ok=True)

    def distribution_plots(self):
        df = self.full_data[self.full_data['age-bias'] == self.type_of_approach]
        added = ""
        if self.keep_originals:
            df = df[df['predictors_origin'] != 'preset']
            added = "_with_no_correction"
        plt.close()
        fig, axes = plt.subplots(2, 3, figsize=(25, 10), constrained_layout=True)

        if self.age_bounded:
            if self.healthy_definition == 'orx':
                numeric_columns = ['MAE_bounded', 'MAE_max_bin_bounded']
            else:
                numeric_columns = ['MAE_cole_bounded', 'MAE_max_bin_cole_bounded']
        else:
            if self.healthy_definition == 'orx':
                numeric_columns = ['MAE', 'MAE_max_bin']
            else:
                numeric_columns = ['MAE_cole', 'MAE_max_bin_cole']

        colors = ['#1f77b4', '#ff69b4']
        categorical_columns = ['method', 'training_population', 'gender', 'sampling', 'features_type']
        for ax, cat_col in zip(axes.flatten(), categorical_columns):
            for col, color in zip(numeric_columns, colors):
                sns.boxplot(data=df, x=cat_col, y=col, ax=ax, color=color)
            ax.set_title(f"Distribution by {cat_col}")
            ax.legend(handles=[
                plt.Line2D([0], [0], color=color, lw=4) for color in colors
            ], labels=numeric_columns, title="Numeric Columns")

        plt.savefig(self.directory_results_global + "distribution_plots_" + self.healthy_definition + '_' + self.type_of_approach + added + "_age_bounded_" + str(self.age_bounded) + ".png")
        plt.show()

    def pearson_plots(self):
        df = self.full_data[self.full_data['age-bias'] == self.type_of_approach]
        added = ""
        if self.keep_originals:
            df = df[df['predictors_origin'] != 'preset']
            added = "_with_no_correction"

        plt.close()
        fig, axes = plt.subplots(2, 3, figsize=(25, 10), constrained_layout=True)

        if self.age_bounded:
            if self.healthy_definition == 'orx':
                numeric_columns = ['Pearson_chronological_brain_age_bounded_healthy_orx', 'Pearson_age_PAD_bounded_healthy_orx']
            else:
                numeric_columns = ['Pearson_chronological_brain_age_bounded_healthy_cole', 'Pearson_age_PAD_bounded_healthy_cole']
        else:
            if self.healthy_definition == 'orx':
                numeric_columns = ['Pearson_chronological_brain_age_healthy_orx', 'Pearson_age_PAD_healthy_orx']
            else:
                numeric_columns = ['Pearson_chronological_brain_age_healthy_cole', 'Pearson_age_PAD_healthy_cole']


        colors = ['#1f77b4', '#ff69b4']
        categorical_columns = ['method', 'training_population', 'gender', 'sampling', 'features_type']
        for ax, cat_col in zip(axes.flatten(), categorical_columns):
            for col, color in zip(numeric_columns, colors):
                sns.boxplot(data=df, x=cat_col, y=col, ax=ax, color=color)#, showfliers=False)
            ax.set_title(f"Distribution by {cat_col}")
            #ax.set_ylim(-0.20, 1.0)
            #ax.set_yticks(np.arange(-0.20, 1.01, 0.10))
            ax.legend(handles=[
                plt.Line2D([0], [0], color=color, lw=4) for color in colors
            ], labels=numeric_columns, title="Numeric Columns")

        plt.savefig(self.directory_results_global + "Pearson_plots_" + self.healthy_definition + '_' + self.type_of_approach + added + "_age_bounded_" + str(self.age_bounded) + ".png")
        plt.show()

    def distribution_plots_conditions(self):
        self.directory_results_condition = self.directory_results_global + 'Condition/' + self.binary_case + '/'
        if not os.path.exists(self.directory_results_condition):
            os.makedirs(self.directory_results_condition, exist_ok=True)

        df = self.full_data[self.full_data['age-bias'] == self.type_of_approach]
        added = ""
        if self.keep_originals:
            df = df[df['predictors_origin'] != 'preset']
            added = "_with_no_correction"

        plt.close()
        fig, axes = plt.subplots(2, 3, figsize=(25, 10), constrained_layout=True)

        numeric_columns = ["auroc_" + self.binary_case]
        colors = ['#1f77b4']
        categorical_columns = ['method', 'training_population', 'gender', 'sampling', 'features_type']
        for ax, cat_col in zip(axes.flatten(), categorical_columns):
            for col, color in zip(numeric_columns, colors):
                sns.boxplot(data=df, x=cat_col, y=col, ax=ax, color=color)
            ax.set_title(f"Distribution by {cat_col}")
            ax.legend(handles=[
                plt.Line2D([0], [0], color=color, lw=4) for color in colors
            ], labels=numeric_columns, title="Numeric Columns")

        plt.savefig(self.directory_results_condition + "binary_plots_" + self.type_of_approach + added + ".png")
        plt.show()
'''
data_locations = ["pareto/Male_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN'].csv",
                  "pareto/Female_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN'].csv"]
directories_results = ['post_analysis/Male/', 'post_analysis/Female/']

for i in [0,1]:
    data_location = data_locations[i]
    directory_results = directories_results[i]
    for age_bounded in [True,False]:
        for type_of_approach in ['uncorrected', 'lange', 'cole', 'zhang']:
            for healthy_definition in ['orx','cole']:
                dp = plots_UKBB(data_location = data_location, directory_results = directory_results,
                                      type_of_approach = type_of_approach, healthy_definition = healthy_definition, age_bounded=age_bounded)
                dp.distribution_plots()
            for binary_case in ["CN_MultipleSclerosis", "CN_Dementia", "CN_MCI", "CN_SMC", "CN_Parkinson", "CN_otherSpinalChord", "CN_noCN", "CN_MultipleSclerosis_bounded", "CN_Dementia_bounded", "CN_MCI_bounded", "CN_SMC_bounded", "CN_Parkinson_bounded", "CN_otherSpinalChord_bounded", "CN_noCN_bounded"]:
                dp = plots_UKBB(data_location = data_location, directory_results = directory_results,
                                      type_of_approach = type_of_approach, binary_case = binary_case, age_bounded= age_bounded)
                dp.distribution_plots_conditions()'''