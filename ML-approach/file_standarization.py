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

class naming():
    def __init__(self, data_location = "pareto/Male_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN'].csv",
                 directory_results = 'male_standardized_results_MC.csv',
                 NC_proposed_naming = 'Metrics_full_list.txt'):
        """
        :param data_location: Location of original database containing full info (demographic and features) to be standardized.
        :param directory_data: Directory where we want to save all generated metadata. Default: Create subfolder named '/data'.
        :param NC_proposed_naming: File containing columns naming by NC.
        :return: Conversion to unified NC-proposed file naming
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.full_data = pd.read_csv(data_location)
        self.directory_results = directory_results

        with open('Metrics_full_list.txt', 'r') as file:
            standard_columns = [line.strip() for line in file if line.strip()]

        info_columns = ['name', 'modelling_strategy', 'method', 'training_population',
                        'used_predictors', 'used_predictors_set_size', 'sampling', 'gender',
                        'predictors_origin', 'preprocessing', 'eval_metric']

        self.columns_to_extract = info_columns + standard_columns


    def unzip_columns(self):

        def flatten_dict_double(d, prefix):
            return {f'{prefix}_{outer_key}_{inner_key}': value
                    for outer_key, inner_dict in d.items()
                    for inner_key, value in inner_dict.items()}

        def create_columns(d, prefix):
            return {f'{prefix}_{key}': value for key, value in d.items()}

        self.full_data['MAE_group'] = self.full_data['MAE_group'].apply(ast.literal_eval)
        self.full_data['mean_PAD_group'] = self.full_data['mean_PAD_group'].apply(ast.literal_eval)
        self.full_data['MAE_db'] = self.full_data['MAE_db'].apply(ast.literal_eval)
        self.full_data['MAE_db_cole'] = self.full_data['MAE_db_cole'].apply(ast.literal_eval)
        self.full_data['MAE_machine'] = self.full_data['MAE_machine'].apply(ast.literal_eval)
        self.full_data['MAE_machine_cole'] = self.full_data['MAE_machine_cole'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer'] = self.full_data['MAE_manufacturer'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer_cole'] = self.full_data['MAE_manufacturer_cole'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type'] = self.full_data['MAE_CN_type'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type_cole'] = self.full_data['MAE_CN_type_cole'].apply(ast.literal_eval)
        self.full_data['MAE_ethnicity'] = self.full_data['MAE_ethnicity'].apply(ast.literal_eval)
        self.full_data['MAE_ethnicity_cole'] = self.full_data['MAE_ethnicity_cole'].apply(ast.literal_eval)
        self.full_data['MAE_education_years'] = self.full_data['MAE_education_years'].apply(ast.literal_eval)
        self.full_data['MAE_education_years_cole'] = self.full_data['MAE_education_years_cole'].apply(ast.literal_eval)

        self.full_data['MAE_db_bounded'] = self.full_data['MAE_db_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_db_cole_bounded'] = self.full_data['MAE_db_cole_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_machine_bounded'] = self.full_data['MAE_machine_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_machine_cole_bounded'] = self.full_data['MAE_machine_cole_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer_bounded'] = self.full_data['MAE_manufacturer_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_manufacturer_cole_bounded'] = self.full_data['MAE_manufacturer_cole_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type_bounded'] = self.full_data['MAE_CN_type_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_CN_type_cole_bounded'] = self.full_data['MAE_CN_type_cole_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_ethnicity_bounded'] = self.full_data['MAE_ethnicity_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_ethnicity_cole_bounded'] = self.full_data['MAE_ethnicity_cole_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_education_years_bounded'] = self.full_data['MAE_education_years_bounded'].apply(ast.literal_eval)
        self.full_data['MAE_education_years_cole_bounded'] = self.full_data['MAE_education_years_cole_bounded'].apply(ast.literal_eval)

        self.full_data['Pearson_chronological_brain_age'] = self.full_data['Pearson_chronological_brain_age'].apply(ast.literal_eval)
        self.full_data['Pearson_age_PAD'] = self.full_data['Pearson_age_PAD'].apply(ast.literal_eval)
        self.full_data['Pearson_chronological_brain_age_bounded'] = self.full_data['Pearson_chronological_brain_age_bounded'].apply(
            ast.literal_eval)
        self.full_data['Pearson_age_PAD_bounded'] = self.full_data['Pearson_age_PAD_bounded'].apply(ast.literal_eval)
        self.full_data['MSE'] = self.full_data['MSE'].apply(ast.literal_eval)
        self.full_data['MSE_bounded'] = self.full_data['MSE_bounded'].apply(ast.literal_eval)
        self.full_data['R-squared'] = self.full_data['R-squared'].apply(ast.literal_eval)
        self.full_data['R-squared_bounded'] = self.full_data['R-squared_bounded'].apply(ast.literal_eval)

        flattened_df = self.full_data['MAE_group'].apply(lambda x: flatten_dict_double(x, 'MAE')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_group']).join(flattened_df)
        new_columns_df = self.full_data['MAE_db'].apply(lambda x: create_columns(x, 'healthy_orx')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_db_cole'].apply(lambda x: create_columns(x, 'healthy_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db_cole']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_ethnicity'].apply(lambda x: create_columns(x, 'healthy_orx')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_ethnicity']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_ethnicity_cole'].apply(lambda x: create_columns(x, 'healthy_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_ethnicity_cole']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_education_years'].apply(lambda x: create_columns(x, 'healthy_orx')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_education_years']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_education_years_cole'].apply(lambda x: create_columns(x, 'healthy_cole')).apply(
            pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_education_years_cole']).join(new_columns_df)

        new_columns_df = self.full_data['MAE_db_bounded'].apply(lambda x: create_columns(x, 'b_healthy_orx')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_db_cole_bounded'].apply(lambda x: create_columns(x, 'b_healthy_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_db_cole_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_ethnicity_bounded'].apply(lambda x: create_columns(x, 'b_healthy_orx')).apply(
            pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_ethnicity_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_ethnicity_cole_bounded'].apply(lambda x: create_columns(x, 'b_healthy_cole')).apply(
            pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_ethnicity_cole_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_education_years_bounded'].apply(lambda x: create_columns(x, 'b_healthy_orx')).apply(
            pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_education_years_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_education_years_cole_bounded'].apply(
            lambda x: create_columns(x, 'b_healthy_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_education_years_cole_bounded']).join(new_columns_df)

        new_columns_df = self.full_data['Pearson_age_PAD'].apply(lambda x: create_columns(x, 'Pearson_age_PAD')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_age_PAD']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_chronological_brain_age'].apply(
            lambda x: create_columns(x, 'Pearson_chronological_brain_age')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_chronological_brain_age']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_age_PAD_bounded'].apply(
            lambda x: create_columns(x, 'Pearson_age_PAD_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_age_PAD_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['Pearson_chronological_brain_age_bounded'].apply(
            lambda x: create_columns(x, 'Pearson_chronological_brain_age_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['Pearson_chronological_brain_age_bounded']).join(new_columns_df)

        new_columns_df = self.full_data['R-squared'].apply(
            lambda x: create_columns(x, 'R-squared')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['R-squared']).join(new_columns_df)
        new_columns_df = self.full_data['R-squared_bounded'].apply(
            lambda x: create_columns(x, 'R-squared_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['R-squared_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MSE'].apply(
            lambda x: create_columns(x, 'MSE')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MSE']).join(new_columns_df)
        new_columns_df = self.full_data['MSE_bounded'].apply(
            lambda x: create_columns(x, 'MSE_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MSE_bounded']).join(new_columns_df)
        flattened_df= self.full_data['mean_PAD_group'].apply(lambda x: flatten_dict_double(x, 'PAD')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['mean_PAD_group']).join(flattened_df)

        new_columns_df = self.full_data['MAE_CN_type_cole_bounded'].apply(
            lambda x: create_columns(x, 'MAE_CN_type_cole_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_CN_type_cole_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_CN_type_bounded'].apply(
            lambda x: create_columns(x, 'MAE_CN_type_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_CN_type_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_manufacturer_cole_bounded'].apply(
            lambda x: create_columns(x, 'MAE_manufacturer_cole_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_manufacturer_cole_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_manufacturer_bounded'].apply(
            lambda x: create_columns(x, 'MAE_manufacturer_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_manufacturer_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_machine_cole_bounded'].apply(
            lambda x: create_columns(x, 'MAE_machine_cole_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_machine_cole_bounded']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_machine_bounded'].apply(
            lambda x: create_columns(x, 'MAE_machine_bounded')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_machine_bounded']).join(new_columns_df)

        new_columns_df = self.full_data['MAE_CN_type_cole'].apply(
            lambda x: create_columns(x, 'MAE_CN_type_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_CN_type_cole']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_CN_type'].apply(
            lambda x: create_columns(x, 'MAE_CN_type')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_CN_type']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_manufacturer_cole'].apply(
            lambda x: create_columns(x, 'MAE_manufacturer_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_manufacturer_cole']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_manufacturer'].apply(
            lambda x: create_columns(x, 'MAE_manufacturer')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_manufacturer']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_machine_cole'].apply(
            lambda x: create_columns(x, 'MAE_machine_cole')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_machine_cole']).join(new_columns_df)
        new_columns_df = self.full_data['MAE_machine'].apply(
            lambda x: create_columns(x, 'MAE_machine')).apply(pd.Series)
        self.full_data = self.full_data.drop(columns=['MAE_machine']).join(new_columns_df)

    def column_renaming_extraction(self):
        column_mapping = {
            "MAE": "MAE_orx_unbounded",
            "MAE_cole": "MAE_cole_unbounded",
            "MAE_bounded": "MAE_orx_bounded",
            "MAE_healthy_cole_55-60": "MAE_cole_age.[55,60]",
            "MAE_healthy_cole_60-65": "MAE_cole_age.[60,65]",
            "MAE_healthy_cole_65-70": "MAE_cole_age.[65,70]",
            "MAE_healthy_cole_70-75": "MAE_cole_age.[70,75]",
            "MAE_healthy_cole_75-80": "MAE_cole_age.[75,80]",
            "MAE_healthy_cole_80-85": "MAE_cole_age.[80,85]",
            "MAE_healthy_orx_55-60": "MAE_orx_age.[55,60]",
            "MAE_healthy_orx_60-65": "MAE_orx_age.[60,65]",
            "MAE_healthy_orx_65-70": "MAE_orx_age.[65,70]",
            "MAE_healthy_orx_70-75": "MAE_orx_age.[70,75]",
            "MAE_healthy_orx_75-80": "MAE_orx_age.[75,80]",
            "MAE_healthy_orx_80-85": "MAE_orx_age.[80,85]",

            "PAD_healthy_cole_55-60": "meanPAD_cole_age.[55,60]",
            "PAD_healthy_cole_60-65": "meanPAD_cole_age.[60,65]",
            "PAD_healthy_cole_65-70": "meanPAD_cole_age.[65,70]",
            "PAD_healthy_cole_70-75": "meanPAD_cole_age.[70,75]",
            "PAD_healthy_cole_75-80": "meanPAD_cole_age.[75,80]",
            "PAD_healthy_cole_80-85": "meanPAD_cole_age.[80,85]",
            "PAD_healthy_orx_55-60": "meanPAD_orx_age.[55,60]",
            "PAD_healthy_orx_60-65": "meanPAD_orx_age.[60,65]",
            "PAD_healthy_orx_65-70": "meanPAD_orx_age.[65,70]",
            "PAD_healthy_orx_70-75": "meanPAD_orx_age.[70,75]",
            "PAD_healthy_orx_75-80": "meanPAD_orx_age.[75,80]",
            "PAD_healthy_orx_80-85": "meanPAD_orx_age.[80,85]",

            "MSE_healthy_orx": "MSE_orx_unbounded",
            "MSE_healthy_cole": "MSE_cole_unbounded",
            "MSE_bounded_healthy_orx": "MSE_orx_bounded",
            "MSE_bounded_healthy_cole": "MSE_cole_bounded",

            "R-squared_healthy_orx": "Rsquared_orx_unbounded",
            "R-squared_healthy_cole": "Rsquared_cole_unbounded",
            "R-squared_bounded_healthy_orx": "Rsquared_orx_bounded",
            "R-squared_bounded_healthy_cole": "Rsquared_cole_bounded",

            "Pearson_age_PAD_healthy_orx": "corr[PAD,CA]_orx_unbounded",
            "Pearson_age_PAD_healthy_cole": "corr[PAD,CA]_cole_unbounded",
            "Pearson_chronological_brain_age_healthy_orx": "corr[BA,CA]_orx_unbounded",
            "Pearson_chronological_brain_age_healthy_cole": "corr[BA,CA]_cole_unbounded",
            "Pearson_age_PAD_bounded_healthy_orx": "corr[PAD,CA]_orx_bounded",
            "Pearson_age_PAD_bounded_healthy_cole": "corr[PAD,CA]_cole_bounded",
            "Pearson_chronological_brain_age_bounded_healthy_orx": "corr[BA,CA]_orx_bounded",
            "Pearson_chronological_brain_age_bounded_healthy_cole": "corr[BA,CA]_cole_bounded",

            "healthy_orx_UKBB": "MAE_orx_unbounded_db.UKBB",
            "healthy_orx_ADNI": "MAE_orx_unbounded_db.ADNI",
            "healthy_orx_NACC": "MAE_orx_unbounded_db.NACC",
            "healthy_cole_UKBB": "MAE_cole_unbounded_db.UKBB",
            "healthy_cole_ADNI": "MAE_cole_unbounded_db.ADNI",
            "healthy_cole_NACC": "MAE_cole_unbounded_db.NACC",
            "b_healthy_orx_UKBB": "MAE_orx_bounded_db.UKBB",
            "b_healthy_orx_ADNI": "MAE_orx_bounded_db.ADNI",
            "b_healthy_orx_NACC": "MAE_orx_bounded_db.NACC",
            "b_healthy_cole_UKBB": "MAE_cole_bounded_db.UKBB",
            "b_healthy_cole_ADNI": "MAE_cole_bounded_db.ADNI",
            "b_healthy_cole_NACC": "MAE_cole_bounded_db.NACC",

            "MAE_manufacturer_cole_bounded_GE": "MAE_cole_bounded_manufacturer.GE",
            "MAE_manufacturer_cole_bounded_Philips": "MAE_cole_bounded_manufacturer.Philips",
            "MAE_manufacturer_cole_bounded_Siemens": "MAE_cole_bounded_manufacturer.Siemens",
            "MAE_manufacturer_cole_GE": "MAE_cole_unbounded_manufacturer.GE",
            "MAE_manufacturer_cole_Philips": "MAE_cole_unbounded_manufacturer.Philips",
            "MAE_manufacturer_cole_Siemens": "MAE_cole_unbounded_manufacturer.Siemens",
            "MAE_manufacturer_bounded_GE": "MAE_orx_bounded_manufacturer.GE",
            "MAE_manufacturer_bounded_Philips": "MAE_orx_bounded_manufacturer.Philips",
            "MAE_manufacturer_bounded_Siemens": "MAE_orx_bounded_manufacturer.Siemens",
            "MAE_manufacturer_GE": "MAE_orx_unbounded_manufacturer.GE",
            "MAE_manufacturer_Philips": "MAE_orx_unbounded_manufacturer.Philips",
            "MAE_manufacturer_Siemens": "MAE_orx_unbounded_manufacturer.Siemens",

            'MAE_machine_cole_bounded_Achieva': 'MAE_cole_bounded_machine.Achieva',
            'MAE_machine_cole_bounded_Allegra': 'MAE_cole_bounded_machine.Allegra',
            'MAE_machine_cole_bounded_Avanto': 'MAE_cole_bounded_machine.Avanto',
            'MAE_machine_cole_bounded_Biograph_mMR': 'MAE_cole_bounded_machine.Biograph.mMR',
            'MAE_machine_cole_bounded_DISCOVERY MR750': 'MAE_cole_bounded_machine.DISCOVERY.MR750',
            'MAE_machine_cole_bounded_Espree': 'MAE_cole_bounded_machine.Espree',
            'MAE_machine_cole_bounded_GEMINI': 'MAE_cole_bounded_machine.GEMINI',
            'MAE_machine_cole_bounded_Gyroscan NT': 'MAE_cole_bounded_machine.Gyroscan.NT',
            'MAE_machine_cole_bounded_Ingenia': 'MAE_cole_bounded_machine.Ingenia',
            'MAE_machine_cole_bounded_Ingenuity': 'MAE_cole_bounded_machine.Ingenuity',
            'MAE_machine_cole_bounded_Intera': 'MAE_cole_bounded_machine.Intera',
            'MAE_machine_cole_bounded_NUMARIS/4': 'MAE_cole_bounded_machine.NUMARIS/4',
            'MAE_machine_cole_bounded_Prisma': 'MAE_cole_bounded_machine.Prisma',
            'MAE_machine_cole_bounded_SIGNA': 'MAE_cole_bounded_machine.SIGNA',
            'MAE_machine_cole_bounded_Skyra': 'MAE_cole_bounded_machine.Skyra',
            'MAE_machine_cole_bounded_Sonata': 'MAE_cole_bounded_machine.Sonata',
            'MAE_machine_cole_bounded_Symphony': 'MAE_cole_bounded_machine.Symphony',
            'MAE_machine_cole_bounded_TrioTim': 'MAE_cole_bounded_machine.TrioTim',
            'MAE_machine_cole_bounded_Verio': 'MAE_cole_bounded_machine.Verio',
            'MAE_machine_cole_Achieva': 'MAE_cole_unbounded_machine.Achieva',
            'MAE_machine_cole_Allegra': 'MAE_cole_unbounded_machine.Allegra',
            'MAE_machine_cole_Avanto': 'MAE_cole_unbounded_machine.Avanto',
            'MAE_machine_cole_Biograph_mMR': 'MAE_cole_unbounded_machine.Biograph.mMR',
            'MAE_machine_cole_DISCOVERY MR750': 'MAE_cole_unbounded_machine.DISCOVERY.MR750',
            'MAE_machine_cole_Espree': 'MAE_cole_unbounded_machine.Espree',
            'MAE_machine_cole_GEMINI': 'MAE_cole_unbounded_machine.GEMINI',
            'MAE_machine_cole_Gyroscan_NT': 'MAE_cole_unbounded_machine.Gyroscan.NT',
            'MAE_machine_cole_Ingenia': 'MAE_cole_unbounded_machine.Ingenia',
            'MAE_machine_cole_Ingenuity': 'MAE_cole_unbounded_machine.Ingenuity',
            'MAE_machine_cole_Intera': 'MAE_cole_unbounded_machine.Intera',
            'MAE_machine_cole_NUMARIS/4': 'MAE_cole_unbounded_machine.NUMARIS/4',
            'MAE_machine_cole_Prisma': 'MAE_cole_unbounded_machine.Prisma',
            'MAE_machine_cole_SIGNA': 'MAE_cole_unbounded_machine.SIGNA',
            'MAE_machine_cole_Skyra': 'MAE_cole_unbounded_machine.Skyra',
            'MAE_machine_cole_Sonata': 'MAE_cole_unbounded_machine.Sonata',
            'MAE_machine_cole_Symphony': 'MAE_cole_unbounded_machine.Symphony',
            'MAE_machine_cole_TrioTim': 'MAE_cole_unbounded_machine.TrioTim',
            'MAE_machine_cole_Verio': 'MAE_cole_unbounded_machine.Verio',
            'MAE_machine_bounded_Achieva': 'MAE_orx_bounded_machine.Achieva',
            'MAE_machine_bounded_Allegra': 'MAE_orx_bounded_machine.Allegra',
            'MAE_machine_bounded_Avanto': 'MAE_orx_bounded_machine.Avanto',
            'MAE_machine_bounded_Biograph_mMR': 'MAE_orx_bounded_machine.Biograph.mMR',
            'MAE_machine_bounded_DISCOVERY MR750': 'MAE_orx_bounded_machine.DISCOVERY.MR750',
            'MAE_machine_bounded_Espree': 'MAE_orx_bounded_machine.Espree',
            'MAE_machine_bounded_GEMINI': 'MAE_orx_bounded_machine.GEMINI',
            'MAE_machine_bounded_Gyroscan_NT': 'MAE_orx_bounded_machine.Gyroscan.NT',
            'MAE_machine_bounded_Ingenia': 'MAE_orx_bounded_machine.Ingenia',
            'MAE_machine_bounded_Ingenuity': 'MAE_orx_bounded_machine.Ingenuity',
            'MAE_machine_bounded_Intera': 'MAE_orx_bounded_machine.Intera',
            'MAE_machine_bounded_NUMARIS/4': 'MAE_orx_bounded_machine.NUMARIS/4',
            'MAE_machine_bounded_Prisma': 'MAE_orx_bounded_machine.Prisma',
            'MAE_machine_bounded_SIGNA': 'MAE_orx_bounded_machine.SIGNA',
            'MAE_machine_bounded_Skyra': 'MAE_orx_bounded_machine.Skyra',
            'MAE_machine_bounded_Sonata': 'MAE_orx_bounded_machine.Sonata',
            'MAE_machine_bounded_Symphony': 'MAE_orx_bounded_machine.Symphony',
            'MAE_machine_bounded_TrioTim': 'MAE_orx_bounded_machine.TrioTim',
            'MAE_machine_bounded_Verio': 'MAE_orx_bounded_machine.Verio',
            'MAE_machine_Achieva': 'MAE_orx_unbounded_machine.Achieva',
            'MAE_machine_Allegra': 'MAE_orx_unbounded_machine.Allegra',
            'MAE_machine_Avanto': 'MAE_orx_unbounded_machine.Avanto',
            'MAE_machine_Biograph_mMR': 'MAE_orx_unbounded_machine.Biograph.mMR',
            'MAE_machine_DISCOVERY MR750': 'MAE_orx_unbounded_machine.DISCOVERY.MR750',
            'MAE_machine_Espree': 'MAE_orx_unbounded_machine.Espree',
            'MAE_machine_GEMINI': 'MAE_orx_unbounded_machine.GEMINI',
            'MAE_machine_Gyroscan_NT': 'MAE_orx_unbounded_machine.Gyroscan.NT',
            'MAE_machine_Ingenia': 'MAE_orx_unbounded_machine.Ingenia',
            'MAE_machine_Ingenuity': 'MAE_orx_unbounded_machine.Ingenuity',
            'MAE_machine_Intera': 'MAE_orx_unbounded_machine.Intera',
            'MAE_machine_NUMARIS/4': 'MAE_orx_unbounded_machine.NUMARIS/4',
            'MAE_machine_Prisma': 'MAE_orx_unbounded_machine.Prisma',
            'MAE_machine_SIGNA': 'MAE_orx_unbounded_machine.SIGNA',
            'MAE_machine_Skyra': 'MAE_orx_unbounded_machine.Skyra',
            'MAE_machine_Sonata': 'MAE_orx_unbounded_machine.Sonata',
            'MAE_machine_Symphony': 'MAE_orx_unbounded_machine.Symphony',
            'MAE_machine_TrioTim': 'MAE_orx_unbounded_machine.TrioTim',
            'MAE_machine_Verio': 'MAE_orx_unbounded_machine.Verio',

            'b_healthy_cole_Asian': 'MAE_cole_bounded_ethnicity.Asian',
            'b_healthy_cole_Black': 'MAE_cole_bounded_ethnicity.Black',
            'b_healthy_cole_Caribbean': 'MAE_cole_bounded_ethnicity.Caribbean',
            'b_healthy_cole_Chinese': 'MAE_cole_bounded_ethnicity.Chinese',
            'b_healthy_cole_Mixed': 'MAE_cole_bounded_ethnicity.Mixed',
            'b_healthy_cole_NativeAmerican': 'MAE_cole_bounded_ethnicity.NativeAmerican',
            'b_healthy_cole_Other': 'MAE_cole_bounded_ethnicity.Other',
            'b_healthy_cole_White': 'MAE_cole_bounded_ethnicity.White',
            'healthy_cole_Asian': 'MAE_cole_unbounded_ethnicity.Asian',
            'healthy_cole_Black': 'MAE_cole_unbounded_ethnicity.Black',
            'healthy_cole_Caribbean': 'MAE_cole_unbounded_ethnicity.Caribbean',
            'healthy_cole_Chinese': 'MAE_cole_unbounded_ethnicity.Chinese',
            'healthy_cole_Mixed': 'MAE_cole_unbounded_ethnicity.Mixed',
            'healthy_cole_NativeAmerican': 'MAE_cole_unbounded_ethnicity.NativeAmerican',
            'healthy_cole_Other': 'MAE_cole_unbounded_ethnicity.Other',
            'healthy_cole_White': 'MAE_cole_unbounded_ethnicity.White',
            'b_healthy_orx_Asian': 'MAE_orx_bounded_ethnicity.Asian',
            'b_healthy_orx_Black': 'MAE_orx_bounded_ethnicity.Black',
            'b_healthy_orx_Caribbean': 'MAE_orx_bounded_ethnicity.Caribbean',
            'b_healthy_orx_Chinese': 'MAE_orx_bounded_ethnicity.Chinese',
            'b_healthy_orx_Mixed': 'MAE_orx_bounded_ethnicity.Mixed',
            'b_healthy_orx_NativeAmerican': 'MAE_orx_bounded_ethnicity.NativeAmerican',
            'b_healthy_orx_Other': 'MAE_orx_bounded_ethnicity.Other',
            'b_healthy_orx_White': 'MAE_orx_bounded_ethnicity.White',
            'healthy_orx_Asian': 'MAE_orx_unbounded_ethnicity.Asian',
            'healthy_orx_Black': 'MAE_orx_unbounded_ethnicity.Black',
            'healthy_orx_Caribbean': 'MAE_orx_unbounded_ethnicity.Caribbean',
            'healthy_orx_Chinese': 'MAE_orx_unbounded_ethnicity.Chinese',
            'healthy_orx_Mixed': 'MAE_orx_unbounded_ethnicity.Mixed',
            'healthy_orx_NativeAmerican': 'MAE_orx_unbounded_ethnicity.NativeAmerican',
            'healthy_orx_Other': 'MAE_orx_unbounded_ethnicity.Other',
            'healthy_orx_White': 'MAE_orx_unbounded_ethnicity.White',
            'MAE_CN_type_cole_bounded_CN_0': 'MAE_cole_bounded_CN.0',
            'MAE_CN_type_cole_bounded_CN_5': 'MAE_cole_bounded_CN.5',
            'MAE_CN_type_cole_bounded_CN_10': 'MAE_cole_bounded_CN.10',
            'MAE_CN_type_cole_CN_0': 'MAE_cole_unbounded_CN.0',
            'MAE_CN_type_cole_CN_5': 'MAE_cole_unbounded_CN.5',
            'MAE_CN_type_cole_CN_10': 'MAE_cole_unbounded_CN.10',
            'MAE_CN_type_bounded_CN_0': 'MAE_orx_bounded_CN.0',
            'MAE_CN_type_bounded_CN_5': 'MAE_orx_bounded_CN.5',
            'MAE_CN_type_bounded_CN_10': 'MAE_orx_bounded_CN.10',
            'MAE_CN_type_CN_0': 'MAE_orx_unbounded_CN.0',
            'MAE_CN_type_CN_5': 'MAE_orx_unbounded_CN.5',
            'MAE_CN_type_CN_10': 'MAE_orx_unbounded_CN.10',
            'b_healthy_cole_0-7': 'MAE_cole_bounded_education.[0,7]',
            'b_healthy_cole_8-15': 'MAE_cole_bounded_education.[8,15]',
            'b_healthy_cole_16-30': 'MAE_cole_bounded_education.[16,30]',
            'healthy_cole_0-7': 'MAE_cole_unbounded_education.[0,7]',
            'healthy_cole_8-15': 'MAE_cole_unbounded_education.[8,15]',
            'healthy_cole_16-30': 'MAE_cole_unbounded_education.[16,30]',
            'b_healthy_orx_0-7': 'MAE_orx_bounded_education.[0,7]',
            'b_healthy_orx_8-15': 'MAE_orx_bounded_education.[8,15]',
            'b_healthy_orx_16-30': 'MAE_orx_bounded_education.[16,30]',
            'healthy_orx_0-7': 'MAE_orx_unbounded_education.[0,7]',
            'healthy_orx_8-15': 'MAE_orx_unbounded_education.[8,15]',
            'healthy_orx_16-30': 'MAE_orx_unbounded_education.[16,30]',

            'auroc_brain_age_CN_noCN_orx_bounded': 'aurocBA_bounded_CNvsCondition.NoCN',
            'auroc_brain_age_CN_MultipleSclerosis_orx_bounded': 'aurocBA_bounded_CNvsCondition.MultipleSclerosis',
            'auroc_brain_age_CN_Dementia_orx_bounded': 'aurocBA_bounded_CNvsCondition.Dementia',
            'auroc_brain_age_CN_MCI_orx_bounded': 'aurocBA_bounded_CNvsCondition.MCI',
            'auroc_brain_age_CN_SMC_orx_bounded': 'aurocBA_bounded_CNvsCondition.SMC',
            'auroc_brain_age_CN_Parkinson_orx_bounded': 'aurocBA_bounded_CNvsCondition.Parkinson',
            'auroc_brain_age_CN_otherSpinalChord_orx_bounded': 'aurocBA_bounded_CNvsCondition.OtherSpinalChord',
            'auroc_brain_age_CN_Alzheimer_orx_bounded': 'aurocBA_bounded_CNvsCondition.Alzheimer',
            'auroc_brain_age_CN_Dementia_Alzheimer_orx_bounded': 'aurocBA_bounded_CNvsCondition.Dementia.Alzheimer',
            'auroc_brain_age_CN_otherSpinalChord_Dementia_orx_bounded': 'aurocBA_bounded_CNvsCondition.OtherSpinalChord.Dementia',
            'auroc_brain_age_CN_otherSpinalChord_MultipleSclerosis_orx_bounded': 'aurocBA_bounded_CNvsCondition.OtherSpinalChord.MultipleSclerosis',
            'auroc_brain_age_CN_otherSpinalChord_Parkinson_orx_bounded': 'aurocBA_bounded_CNvsCondition.OtherSpinalChord.Parkinson',
            'auroc_brain_age_CN_Alzheimer_Dementia_orx_bounded': 'aurocBA_bounded_CNvsCondition.Alzheimer.Dementia',
            'auroc_brain_age_CN_noCN_orx': 'aurocBA_unbounded_CNvsCondition.NoCN',
            'auroc_brain_age_CN_MultipleSclerosis_orx': 'aurocBA_unbounded_CNvsCondition.MultipleSclerosis',
            'auroc_brain_age_CN_Dementia_orx': 'aurocBA_unbounded_CNvsCondition.Dementia',
            'auroc_brain_age_CN_MCI_orx': 'aurocBA_unbounded_CNvsCondition.MCI',
            'auroc_brain_age_CN_SMC_orx': 'aurocBA_unbounded_CNvsCondition.SMC',
            'auroc_brain_age_CN_Parkinson_orx': 'aurocBA_unbounded_CNvsCondition.Parkinson',
            'auroc_brain_age_CN_otherSpinalChord_orx': 'aurocBA_unbounded_CNvsCondition.OtherSpinalChord',
            'auroc_brain_age_CN_Alzheimer_orx': 'aurocBA_unbounded_CNvsCondition.Alzheimer',
            'auroc_brain_age_CN_Dementia_Alzheimer_orx': 'aurocBA_unbounded_CNvsCondition.Dementia.Alzheimer',
            'auroc_brain_age_CN_otherSpinalChord_Dementia_orx': 'aurocBA_unbounded_CNvsCondition.OtherSpinalChord.Dementia',
            'auroc_brain_age_CN_otherSpinalChord_MultipleSclerosis_orx': 'aurocBA_unbounded_CNvsCondition.OtherSpinalChord.MultipleSclerosis',
            'auroc_brain_age_CN_otherSpinalChord_Parkinson_orx': 'aurocBA_unbounded_CNvsCondition.OtherSpinalChord.Parkinson',
            'auroc_brain_age_CN_Alzheimer_Dementia_orx': 'aurocBA_unbounded_CNvsCondition.Alzheimer.Dementia',

            'auroc_PAD_CN_noCN_orx_bounded': 'aurocPAD_bounded_CNvsCondition.NoCN',
            'auroc_PAD_CN_MultipleSclerosis_orx_bounded': 'aurocPAD_bounded_CNvsCondition.MultipleSclerosis',
            'auroc_PAD_CN_Dementia_orx_bounded': 'aurocPAD_bounded_CNvsCondition.Dementia',
            'auroc_PAD_CN_MCI_orx_bounded': 'aurocPAD_bounded_CNvsCondition.MCI',
            'auroc_PAD_CN_SMC_orx_bounded': 'aurocPAD_bounded_CNvsCondition.SMC',
            'auroc_PAD_CN_Parkinson_orx_bounded': 'aurocPAD_bounded_CNvsCondition.Parkinson',
            'auroc_PAD_CN_otherSpinalChord_orx_bounded': 'aurocPAD_bounded_CNvsCondition.OtherSpinalChord',
            'auroc_PAD_CN_Alzheimer_orx_bounded': 'aurocPAD_bounded_CNvsCondition.Alzheimer',
            'auroc_PAD_CN_Dementia_Alzheimer_orx_bounded': 'aurocPAD_bounded_CNvsCondition.Dementia.Alzheimer',
            'auroc_PAD_CN_otherSpinalChord_Dementia_orx_bounded': 'aurocPAD_bounded_CNvsCondition.OtherSpinalChord.Dementia',
            'auroc_PAD_CN_otherSpinalChord_MultipleSclerosis_orx_bounded': 'aurocPAD_bounded_CNvsCondition.OtherSpinalChord.MultipleSclerosis',
            'auroc_PAD_CN_otherSpinalChord_Parkinson_orx_bounded': 'aurocPAD_bounded_CNvsCondition.OtherSpinalChord.Parkinson',
            'auroc_PAD_CN_Alzheimer_Dementia_orx_bounded': 'aurocPAD_bounded_CNvsCondition.Alzheimer.Dementia',
            'auroc_PAD_CN_noCN_orx': 'aurocPAD_unbounded_CNvsCondition.NoCN',
            'auroc_PAD_CN_MultipleSclerosis_orx': 'aurocPAD_unbounded_CNvsCondition.MultipleSclerosis',
            'auroc_PAD_CN_Dementia_orx': 'aurocPAD_unbounded_CNvsCondition.Dementia',
            'auroc_PAD_CN_MCI_orx': 'aurocPAD_unbounded_CNvsCondition.MCI',
            'auroc_PAD_CN_SMC_orx': 'aurocPAD_unbounded_CNvsCondition.SMC',
            'auroc_PAD_CN_Parkinson_orx': 'aurocPAD_unbounded_CNvsCondition.Parkinson',
            'auroc_PAD_CN_otherSpinalChord_orx': 'aurocPAD_unbounded_CNvsCondition.OtherSpinalChord',
            'auroc_PAD_CN_Alzheimer_orx': 'aurocPAD_unbounded_CNvsCondition.Alzheimer',
            'auroc_PAD_CN_Dementia_Alzheimer_orx': 'aurocPAD_unbounded_CNvsCondition.Dementia.Alzheimer',
            'auroc_PAD_CN_otherSpinalChord_Dementia_orx': 'aurocPAD_unbounded_CNvsCondition.OtherSpinalChord.Dementia',
            'auroc_PAD_CN_otherSpinalChord_MultipleSclerosis_orx': 'aurocPAD_unbounded_CNvsCondition.OtherSpinalChord.MultipleSclerosis',
            'auroc_PAD_CN_otherSpinalChord_Parkinson_orx': 'aurocPAD_unbounded_CNvsCondition.OtherSpinalChord.Parkinson',
            'auroc_PAD_CN_Alzheimer_Dementia_orx': 'aurocPAD_unbounded_CNvsCondition.Alzheimer.Dementia'
        }

        # Iterate over the column mappings to make the changes
        for old_col, new_col in column_mapping.items():
            if old_col in self.full_data.columns:
                # Rename the column if it exists
                self.full_data.rename(columns={old_col: new_col}, inplace=True)
            else:
                # Create a new column filled with NaNs if the old column doesn't exist
                self.full_data[new_col] = np.nan
        df = self.full_data[self.columns_to_extract]
        df.to_csv(self.directory_results, index=False)

        columns_only_nans = df.columns[df.isna().all()].tolist()
        print("Columns with only NaN values:")
        print(columns_only_nans)


for db in ['UKBB', 'ADNI', 'NACC', 'None']:
    data_locations = ["pareto/"+db+"_Male_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN_orx'].csv",
                      "pareto/"+db+"_Female_all_T1w_feats_pval_age-biased/analyzed_cases_['MAE', 'MAE_max_bin', 'auroc_CN_noCN_orx'].csv"]
    directories_results = [db+'_male_standardized_results_MC.csv', db+'_female_standardized_results_MC.csv']

    for i in [0,1]:
        data_location = data_locations[i]
        directory_results = directories_results[i]
        dp = naming(data_location = data_location, directory_results = directory_results)
        dp.unzip_columns()
        dp.column_renaming_extraction()
