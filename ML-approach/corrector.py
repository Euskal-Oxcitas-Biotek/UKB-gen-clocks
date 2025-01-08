import numpy as np
import pandas as pd
import linear_model as cl
import json
import ast
import os


class retrain():
    def __init__(self, X = None, save_metrics = True,
                 directory_metrics = 'metrics_train/', directory_models = 'models/',
                 p_val_threshold = 0.05, run_id = 1, type_of_models_to_correct = ['ols']):
        """
        :param X: Training set
        :param y: Target (age) in training set.
        :param fit_intercept: Boolean (True, False) to decide whether or not to use intercept in the customized regression. Default: True.
        :param save_metrics: Boolean (True, False) to decide whether or not to save the training metrics. Default: True.
        :param directory_metrics: Directory where training metrics should be stored. Default: 'metrics_train/'.
        :param directory_models: Directory where generated models should be stored. Default: 'metrics_models/'.
        :param p_val_threshold: Threshold used to reject null hypothesis of .p_val_test. Default: 0.05.
        :param run_id: Execution id number which is used to look into corresponding history of settings combinations explored/corrected file. Default: 1.
        """
        self.X = X
        self.save_metrics = save_metrics
        self.directory_metrics = directory_metrics
        self.directory_models = directory_models
        self.p_val_threshold = p_val_threshold
        self.type_of_models_to_correct = type_of_models_to_correct
        self.amount_corrected = 0
        #---------------------------------------------------------------------------------------------------------------
        self.run_id = run_id
        self.historic_filepath = 'historic_register_' + str(self.run_id) + '.json' #Extract history to only perform computations to have not been done previously
        with open(self.historic_filepath, 'r') as file:
            data_history = json.load(file)
            self.examined_combinations = data_history['examined_combinations']
            self.corrected_ids = data_history['corrected_ids']
            self.id = data_history['id']
            self.tested = data_history['tested']
        #---------------------------------------------------------------------------------------------------------------


    def models_to_correct(self):
        """
        :return: Function to decide pre-trained models to be tested.
        """
        dataframes = []
        for filename in os.listdir(self.directory_metrics):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.directory_metrics, filename)
                df = pd.read_csv(file_path)
                if df['method'].values[0] in self.type_of_models_to_correct:
                    if (not df['p-values_success'].values[0]) and (str(df['age-bias'].iloc[0])=='nan'):
                        predictors = [elem for elem in ast.literal_eval(df['predictors'].values[0]) if
                                      elem not in ast.literal_eval(df['faulty_coefficients'].values[0])]
                        if len(predictors) > 0:
                            if df['name'].values[0] not in self.corrected_ids:
                                custom_linear = cl.LMB(gender = df['gender'].values[0], predictors = predictors,
                                                       training_population = df['training_population'].values[0], method = df['method'].values[0], sampling = df['sampling'].values[0],
                                                       correction_of=df['name'].values[0], directory_metrics = self.directory_metrics, directory_models = self.directory_models,
                                                       p_val_threshold = self.p_val_threshold, run_id = self.run_id, database_origin = df['database_origin'].values[0], T1W = ast.literal_eval(df['T1W'].values[0]),
                                                       age_limits = ast.literal_eval(df['age_limits'].values[0]), update_history_combinations = False)
                                custom_linear.fit(self.X)
                                self.corrected_ids.append(df['name'].values[0])
                                self.id += 1
                                self.update_history()
                                self.amount_corrected += 1
                            else:
                                print(df['name'].values[0] + ' already corrected.')

    def update_history(self):
        """
        :return: Update associated .json file with already executed models train configuration/corrected models/tested models info
        """
        data = {'examined_combinations' : self.examined_combinations, 'corrected_ids' : self.corrected_ids,'id' : self.id + 1, 'tested' : self.tested}
        with open(self.historic_filepath, 'w') as file:
            json.dump(data, file)
