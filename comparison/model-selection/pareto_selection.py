import numpy as np
import pandas as pd
import os


class model_selection():
    def __init__(self, data_location = "UKBB_male_standardized_results_MC_ML.csv",
                 directory_results = 'pareto_fronts/',
                 features_pareto = ['MAE_orx_unbounded','max_MAE_bin_orx_unbounded','auroc_unbounded_CNvsCondition.NoCN'],
                 non_pareto_features = ['Model_name', 'modelling_strategy', 'Description', 'Age_bias_correction', 'features_type', 'Training_healthy_group', 'Oversampling', 'Training_sex',
                                        'used_predictors','used_predictors_set_size'],
                 save_csv = False, csv_name = None):
        """
        :param data_location: Location of database containing the metrics of the models to be analyzed.
        :param directory_results: Directory where we want to save Pareto fronts. Default: Create subfolder named 'pareto_fronts/'.
        :param features_pareto: Feature/metrics to consider when determining Pareto front. Default: ['MAE_orx_unbounded','max_MAE_bin_orx_unbounded','auroc_unbounded_CNvsCondition.NoCN']
        :param non_pareto_features: Non-numeric features/columns from the dataframe at data_location used to identify the models in the file with pareto front. Default:
               ['name', 'modelling_strategy', 'method', 'age-bias', 'features_type', 'training_population', 'sampling', 'gender']
        :param save_csv: Boolean to specify whether or not to save outcome as .csv
        :param csv_name: If save_csv = True, csv_name is the name of file containing the Pareto front, common option:
               'pareto_' + str(self.features_pareto) + 'data['+self.data_location[:-4]+']_age_biased_exclusion_'+str(self.age_bias_exclusion)+'.csv'
        :return: Pareto front
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.data_location = data_location
        #----------------------------------------------
        # 0.1. The input data_location can be either a string with the directory where the dataframe is stored or the dataframe.
        if isinstance(data_location, str):
            self.full_data = pd.read_csv(data_location)
        else:
            self.full_data = data_location
        #-----------------------------------------------
        self.features_pareto = features_pareto
        self.non_pareto_features = non_pareto_features
        self.save_csv = save_csv
        self.csv_name = csv_name
        self.directory_results = directory_results
        if not os.path.exists(self.directory_results):
            os.makedirs(self.directory_results, exist_ok=True)

    def fit(self):
        """
        :return: Pareto front-based model selector
        """
        missing_columns = []  # List to track missing columns
        # -------------------------------------------------------------------------------
        # STEP 0: Function to verify whether a model is dominated by the performance of a different candidate model.
        def dominates(row1, row2):
            return all(a <= b for a, b in zip(row1, row2)) and any(a < b for a, b in zip(row1, row2))
        # -------------------------------------------------------------------------------
        # STEP 1: Keep rows with no missing values in 'features_pareto'
        filtered_data = self.full_data.dropna(subset=self.features_pareto)
        # 1.1. Check if any rows remain after filtering
        if filtered_data.empty:
            # If no rows are left, remove columns with any missing values in 'features_pareto'
            columns_to_keep = self.full_data[self.features_pareto].dropna(axis=1).columns
            # Ensure 'Model_name' and other important non-pareto features are kept
            columns_to_keep = list(columns_to_keep) + self.non_pareto_features
            self.full_data = self.full_data[columns_to_keep]
            print(f"After filtering, no rows remained. Columns kept (without missing values): {list(columns_to_keep)}")
        else:# 1.2. If rows are retained, use the filtered data
            rows_dropped = len(self.full_data) - len(filtered_data)
            dropped_rows = self.full_data.loc[~self.full_data['Model_name'].isin(filtered_data['Model_name']), 'Model_name']
            print(f"After filtering, {rows_dropped} rows were dropped.")
            print(f"Model_name entries of the dropped rows: {dropped_rows.tolist()}")
            self.full_data = filtered_data
        # -------------------------------------------------------------------------------
        # STEP 2: Generate Pareto front maximizing all performance metrics.
        self.correct_sign() # change signs of those features that must be maximized instead of minimized
        pareto_front = []
        # print(self.full_data.columns)
        for _, row1 in self.full_data.iterrows():
            is_dominated = False
            for _, row2 in self.full_data.iterrows():
                try:
                    # Try to access the features for the Pareto comparison
                    if row1 is not row2 and dominates(row2[self.features_pareto], row1[self.features_pareto]):
                        is_dominated = True
                        break
                except KeyError as e:
                    # If a KeyError occurs, add the missing column to the list and continue
                    missing_columns.append(str(e))
                    continue
            if not is_dominated:
                pareto_front.append(row1['Model_name'])

        self.correct_sign() # revert previous sign change
        self.full_data['pareto'] = self.full_data.apply(lambda row: row['Model_name'] in pareto_front, axis=1)

        # Get the columns that exist in self.full_data[self.full_data['pareto']]
        existing_columns = self.full_data[self.full_data['pareto']].columns

        # Combine the columns you're interested in
        required_columns = self.non_pareto_features + self.features_pareto

        # Find missing columns
        missing_columns = [col for col in required_columns if col not in existing_columns]

        # Print existing and missing columns
        print(f"Existing columns in the DataFrame: {list(existing_columns)}")
        if missing_columns:
            print(
                f"Warning: The following columns were missing and will be dropped from the required columns: {missing_columns}")
            # Remove the missing columns from the required columns
            required_columns = [col for col in required_columns if col not in missing_columns]
        else:
            print("All required columns exist in the DataFrame.")

        self.pareto_front = self.full_data[self.full_data['pareto']][required_columns]

        if self.save_csv:
            self.pareto_front.to_csv(os.path.join(self.directory_results , self.csv_name), index=False)

        # Print out the list of missing columns after the loop finishes
        if missing_columns:
            print("The following columns were missing and skipped:", set(missing_columns))

    def correct_sign(self):
        """
        :return: Function to revert sign of features/metrics that must be maximized instead of minimized
        """
        for col in self.features_pareto:
            if col.startswith('auroc') or col.startswith('corr[BA,CA]') or col.startswith('Rsquared'):
                print(f"Processing column: {col}")
                try:
                    # Try to flip the sign of the column in the DataFrame
                    self.full_data[col] *= -1
                except KeyError:
                    # If the column does not exist, print a warning and continue
                    print(f"Warning: Column '{col}' does not exist in the DataFrame and will be skipped.")
                    continue