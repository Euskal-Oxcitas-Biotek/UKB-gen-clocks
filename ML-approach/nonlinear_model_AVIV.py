import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from flaml import AutoML
from tpot import TPOTRegressor
import joblib
import json
import os
import pickle
import age_level_correction
import lightgbm as lgb
import xgboost as xgb
from hgboost import hgboost


class LMB(BaseEstimator, RegressorMixin):
    def __init__(self, method = 'topt', predictors = None, training_population = 'orx', age_limits = [55, 85],
                 bins_length = 5, sampling = None, bins_lenght_oversample = 5, T1W = {'ctx_feats' : 'Merged', 'LR_feats' : 'Merged', 'CC_feats' : True, 'extra_feats' : True},
                 save_metrics = True, directory_data = 'data/', directory_metrics = 'metrics_train/',
                 directory_models = 'models/', gender = 'All', database_origin = None, run_id = 0,
                 update_history_combinations = True, high_pairwise_correlation_removal = False, evaluation_metric = 'mae'):
        """
        :param method: Linear-based model selected to model the brain clock. Options: ['ols', 'lasso', 'ridge', 'elastic-net']. Default: 'ols'.
        :param predictors: Set of volumetric features to be used. Set it to None if the set of predictors is to be generated using ctx_feats, LR_feats, CC_feats and extra_feats. Default: None.
        :param age_limits: Age interval in years (list: [a,b]) denoting the age bracket considered during training/testing of developed models. Default: a=55,b=85.
        :param bins_length: Length of bins (in years) to be used to compute the training performance metrics. Default: 5.
        :param sampling: Sampling technique to be considered when preprocessing training data. Options: [None, 'smote', 'adasyn', 'resample']. Option: None.
        :param bins_length_oversample: Length of bins (in years) to be used to generate new samples via the selected sampling approach. Default: 5.
        :param T1W: Defining the initial set of T1W volumetric features: 1) ctx_feats/LR_feats: Setting to use for ctx and LR volumetric features.
               Options: 'All', 'Merged' (sum per volume), None (neglect all). Default: 'Merged'. 2) CC_feats/extra_feats: Setting to use for CC and extra volumetric features.
               Options: True (all), False (neglect all). Default: True.
        :param save_metrics: Boolean (True, False) to decide whether or not to save the training metrics. Default: True.
        :param directory_metrics: Directory where training metrics should be stored. Default: 'metrics_train/'.
        :param directory_models: Directory where generated models should be stored. Default: 'metrics_models/'.
        :param gender: Gender to be used to build models. Options: 'male', 'female', 'all' (if 'all' is used, 'PTGENDER' is considered a predictor, otherwise neglected). Default: 'all'.
        :param database_origin: Name of database used to train the model. Default: None.
        :param id: Integer used to generate a unique name for the current model. Default: 1.
        :param examined_combinations: Combinations of settings already considered in previously generated models. Default: [].
        :param run_id: Execution id number which is used to look into corresponding history of settings combinations explored/corrected file. Default: 1.
        :param update_history_combinations: Boolean to decide whether or not to save latest analyzed combination of parameters (just disabled when correcting model). Default: True.
        :param high_pairwise_correlation_removal: Boolean to decide whether or not to eliminate highly correlated predictors -|corr|>0.80- (keep the most correlated to target). Default: False.
        :param evaluation_metric: Metric to be used by autoML libraries to select most promising model. Default: 'mae'
        :return: This class is to build the age-bias linear regression models (Beheshti, Cole, Lange)
        """
        self.method = method
        self.evaluation_metric = evaluation_metric
        self.age_limits = age_limits
        self.training_population = training_population
        self.T1W = T1W
        self.sampling = sampling
        self.alpha = {}
        self.beta = {}
        self.save_metrics = save_metrics
        self.directory_data = directory_data
        self.directory_metrics = directory_metrics
        self.directory_models = directory_models
        self.gender = gender
        self.database_origin = database_origin
        self.high_pairwise_correlation_removal = high_pairwise_correlation_removal
        self.update_history_combinations = update_history_combinations
        self.preprocessing = np.nan
        if self.high_pairwise_correlation_removal:
            self.preprocessing = 'pairwise_correlation'
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
        # STEP 0: Constructing initial set of predictors.
        #0.0: Defining family of volumetric features

        ctx_features = ['caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal',
                        'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
                        'middletemporal', 'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis',
                        'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
                        'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal', 'transversetemporal', 'insula']

        LR_features = ['Cerebral-White-Matter', 'Lateral-Ventricle', 'Inf-Lat-Vent', 'Cerebellum-White-Matter', 'Cerebellum-Cortex',
                       'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens-area', 'VentralDC', 'choroid-plexus']

        CC_features = ['CC_Posterior', 'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

        extra_features = ['3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'CSF', 'WM-hypointensities']

        #0.1: Building predictors set using user input on T1W.
        if predictors is None:
            self.predictors = []
            if self.T1W['CC_feats']:
                self.predictors+= CC_features
            if self.T1W['extra_feats']:
                self.predictors+= extra_features
            if self.T1W['ctx_feats'] == 'All':
                self.predictors += ['ctx-rh-' + col for col in ctx_features]
                self.predictors += ['ctx-lh-' + col for col in ctx_features]
            elif self.T1W['ctx_feats'] == 'Merged':
                self.predictors += ctx_features
            if self.T1W['LR_feats'] == 'All':
                self.predictors += ['Right-' + col for col in LR_features]
                self.predictors += ['Left-' + col for col in LR_features]
            elif self.T1W['LR_feats'] == 'Merged':
                self.predictors += LR_features
            if self.gender == 'All':
                self.predictors = self.predictors + ['PTGENDER']
            self.predictors_origin = str(self.T1W)
        else:
            self.predictors = predictors
            self.predictors_origin = 'preset'
        #---------------------------------------------------------------------------------------------------------------
        # STEP 1: Selection of bins limits
        self.bins = self.bins_limits(bins_length)
        if self.sampling is not None:
            self.bins_oversample = self.bins_limits(bins_lenght_oversample)
        #---------------------------------------------------------------------------------------------------------------
        # STEP 2: Create additional directories if needed
        if not os.path.exists(self.directory_models):
            os.makedirs(self.directory_models, exist_ok=True)
        if not os.path.exists(self.directory_metrics):
            os.makedirs(self.directory_metrics, exist_ok=True)

        self.combination = self.method + '_' + str(self.age_limits) + '_' + self.training_population + '_' + str(
            self.sampling) + '_' + self.database_origin + '_' + self.predictors_origin + '_' + self.gender + '_' + str(self.preprocessing) + '_' + self.evaluation_metric

        self.proceed = self.combination not in self.examined_combinations


    def bins_limits(self, bins_length):
        """
        return: Function to generate bins limits according to information provided by user.
        """
        bins = []
        current = self.age_limits[0]
        while current <= self.age_limits[1]:
            bins.append(current)
            current += bins_length
        return bins


    def training_data_preprocessing(self,X):
        """
        :param X: Predictors of training set (Defined by .predictors). Target is assumed to be 'age_at_scan'
        :return: Function to apply all preprocessing steps selected by user over training set-
        """

        # --------------------------------------------------------------------------------------------------------------
        # STEP 1: Prepare data to fit: Type of CN and gender. Feature elimination.
        X = X[X['healthy_' + self.training_population]]
        if self.gender != 'All':
            X = X[X['biological_sex'] == self.gender]
        # --------------------------------------------------------------------------------------------------------------
        # STEP 2: Feature elimination (constants + highly correlated pairwise features)
        self.predictors = [col for col in self.predictors if X[col].nunique() > 1] #eliminate constant cols
        X = X[self.predictors + ['age_at_scan']].dropna()
        y = X['age_at_scan']
        if self.high_pairwise_correlation_removal:
            correlation_threshold = 0.8
            corr_matrix = X.corr().abs()
            to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > correlation_threshold:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_with_target_col1 = corr_matrix.loc['age_at_scan', col1]
                        corr_with_target_col2 = corr_matrix.loc['age_at_scan', col2]
                        # Add the column with the lower correlation with the target to the drop set
                        if corr_with_target_col1 > corr_with_target_col2:
                            to_drop.add(col2)
                        else:
                            to_drop.add(col1)
            self.predictors = [col for col in self.predictors if col not in to_drop]
            predictors = self.predictors + ['age_at_scan']
            X = X[predictors]
            y = X['age_at_scan']
        # --------------------------------------------------------------------------------------------------
        # STEP 3: Apply preselected sampling approach:
        if self.sampling is not None:
            X['bin'] = pd.cut(X['age_at_scan'], bins=self.bins_oversample, right=False, labels=False)
            if self.sampling == 'resample':
                max_samples = X['bin'].value_counts().max()
                oversampled_data = []
                for bin_val, group in X.groupby('bin'):
                    oversampled_group = resample(group, replace=True, n_samples=max_samples, random_state=42)
                    oversampled_data.append(oversampled_group)
                oversampled_df = pd.concat(oversampled_data)
                X = oversampled_df.drop(['age_at_scan','bin'], axis=1)
                y = oversampled_df['age_at_scan']
            elif self.sampling == 'smote':
                y_target = X['bin']
                X_features = X.drop(columns=['bin'])
                smote = SMOTE()
                X_smote, y_smote = smote.fit_resample(X_features, y_target)
                X = X_smote.drop(columns = ['age_at_scan'])
                y = X_smote['age_at_scan']
            elif self.sampling == 'adasyn':
                y_target = X['bin']
                X_features = X.drop(columns=['bin'])
                adasyn = ADASYN(sampling_strategy='minority')
                X_adasyn, y_adasyn = adasyn.fit_resample(X_features, y_target)
                X = X_adasyn.drop(columns = ['age_at_scan'])
                y = X_adasyn['age_at_scan']
        # --------------------------------------------------------------------------------------------------
        return X, y

    def fit(self, X):
        """
        :param X: Predictors of training set (Defined by .predictors). Target is assumed to be 'age_at_scan'
        :return: Fitted linear regression model and corresponding bias terms (Beheshti 1/2, Cole and Lange)
        """
        #--------------------------------------------------------------------------------------------------
        # STEP 1: Prepare data to fit:
        X, y_train = self.training_data_preprocessing(X)
        X_train = X[self.predictors]#self.scaler.fit_transform(X[self.predictors])
        # --------------------------------------------------------------------------------------------------
        # STEP 2: Fitting corresponding model
        if self.method == 'tpot':
            model = TPOTRegressor(
                generations=5,#5
                population_size=10,#10
                verbosity=0,
                random_state=42,
                scoring=self.evaluation_metric
            )
            model.fit(X_train, y_train)
            self.model = model.fitted_pipeline_
            self.modelling_strategy = str(self.model.named_steps)#self.model.fitted_pipeline_.named_steps
        elif self.method == 'flaml':
            model = AutoML()
            model.fit(X_train, y_train, task="regression", time_budget=300, metric=self.evaluation_metric)  # time_budget is in seconds
            self.model = model.model
            self.modelling_strategy = type(self.model).__name__
        elif self.method == 'xgb_hyperopt':
            print('* ' + self.method + ' ' + self.evaluation_metric)
            hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10,
                          random_state=None)
            results = hgb.xgboost_reg(X_train, y_train.values, eval_metric=self.evaluation_metric)

            model = xgb.XGBRegressor(base_score=hgb.model.base_score, booster=hgb.model.booster,
                                      callbacks=hgb.model.callbacks,
                                      colsample_bylevel=hgb.model.colsample_bylevel,
                                      colsample_bynode=hgb.model.colsample_bynode,
                                      colsample_bytree=hgb.model.colsample_bytree,
                                      early_stopping_rounds=hgb.model.early_stopping_rounds,
                                      enable_categorical=hgb.model.enable_categorical,
                                      eval_metric=hgb.model.eval_metric,
                                      feature_types=hgb.model.feature_types,
                                      gamma=hgb.model.gamma, grow_policy=hgb.model.grow_policy,
                                      importance_type=hgb.model.importance_type,
                                      interaction_constraints=hgb.model.interaction_constraints,
                                      learning_rate=hgb.model.learning_rate,
                                      max_bin=hgb.model.max_bin, max_cat_threshold=hgb.model.max_cat_threshold,
                                      max_cat_to_onehot=hgb.model.max_cat_to_onehot,
                                      max_delta_step=hgb.model.max_delta_step, max_depth=hgb.model.max_depth,
                                      max_leaves=hgb.model.max_leaves,
                                      min_child_weight=hgb.model.min_child_weight, missing=hgb.model.missing,
                                      monotone_constraints=hgb.model.monotone_constraints,
                                      n_estimators=hgb.model.n_estimators, n_jobs=hgb.model.n_jobs,
                                      num_parallel_tree=hgb.model.num_parallel_tree,
                                      random_state=hgb.model.random_state,
                                      reg_alpha=hgb.model.reg_alpha, reg_lambda=hgb.model.reg_lambda,
                                      sampling_method=hgb.model.sampling_method,
                                      scale_pos_weight=hgb.model.scale_pos_weight, subsample=hgb.model.subsample,
                                      tree_method=hgb.model.tree_method, objective=hgb.model.objective,
                                      validate_parameters=hgb.model.validate_parameters)
            model.fit(X_train, y_train.values)
            self.model = model
            self.modelling_strategy = self.method
        elif self.method == 'lgb_hyperopt':
            print('* '+self.method+' '+self.evaluation_metric)
            hgb = hgboost(max_eval=250, threshold=0.5, cv=5, test_size=0.2, val_size=0.2, top_cv_evals=10,
                          random_state=None)
            results = hgb.lightboost_reg(X_train, y_train.values, eval_metric=self.evaluation_metric)

            params = {
                'learning_rate': hgb.model.learning_rate,
                'max_depth': hgb.model.max_depth,
                'min_child_weight': hgb.model.min_child_weight,
                'n_estimators': hgb.model.n_estimators,
                'subsample': hgb.model.subsample,
                'verbose': -1
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data)
            self.model = model
            self.modelling_strategy = self.method
        # --------------------------------------------------------------------------------------------------
        # STEP 3: Learn associated age bias terms

        y_pred = self.model.predict(X_train)
        # * Cole correction:
        age_bias_cole = LinearRegression()
        age_bias_cole.fit(np.array(y_train).reshape(-1, 1),y_pred)
        self.alpha['cole'] = age_bias_cole.coef_[0]
        self.beta['cole'] = age_bias_cole.intercept_
        # * Beheshti correction:
        age_bias_beheshti = LinearRegression()
        age_bias_beheshti.fit(np.array(y_train).reshape(-1, 1),y_pred-y_train)
        self.alpha['beheshti'] = age_bias_beheshti.coef_[0]
        self.beta['beheshti'] = age_bias_beheshti.intercept_
        # * Zhang's age-level correction:
        correction = age_level_correction.zhang(CA = y_train, BA = y_pred, age_limits = self.age_limits)
        correction.age_level_bias_training()
        self.zhang_parameters = correction.zhang_parameters
        #---------------------------------------------------------------------------------------------------
        # STEP 4: Get predictors wih non-null coefficients
        non_null_predictors = self.predictors
        # --------------------------------------------------------------------------------------------------
        # STEP 4: If user desires, save training metadata
        if self.save_metrics:
            import train_features
            for bias_type in [None, 'cole', 'lange', 'zhang']: #'beheshti_1', 'beheshti_2'
                self.name = 'model_id_' + str(self.run_id) + '_' + str(self.id)
                y_pred = self.predict(X, bias_type=bias_type, y=y_train)
                #print(bias_type, y_pred)
                metrics = train_features.metrics(model = self, X = X_train, y = y_train, y_pred = self.predict(X, bias_type = bias_type, y = y_train))
                metrics.individual_components()
                if bias_type == None:
                    name = self.name
                else:
                    name = self.name + '_' + bias_type
                p_values = np.nan
                any_bad_p_value = np.nan
                p_value_success = np.nan
                bad_coefficients = np.nan

                df = pd.DataFrame([[name, self.method, self.modelling_strategy, self.training_population, self.predictors, non_null_predictors, len(non_null_predictors),
                                    self.predictors_origin, self.gender, self.evaluation_metric, self.sampling, self.preprocessing, str(self.T1W),
                                    metrics.mse, metrics.mae, metrics.mae_bins, metrics.meanPAD_bins, p_values, p_value_success, bad_coefficients,
                                    bias_type, np.nan, self.database_origin, self.age_limits, self.name]],
                                  columns=['name', 'method', 'modelling_strategy', 'training_population', 'predictors', 'non_null_predictors', 'non_null_preditors_set_size','predictors_origin', 'gender',
                                           'eval_metric','sampling', 'preprocessing','T1W','MSE', 'MAE', 'MAE_group',
                                           'mean_PAD_group', 'p-values','p-values_success', 'faulty_coefficients', 'age-bias', 'correction_of', 'database_origin', 'age_limits', 'save_model'])
                df.to_csv(self.directory_metrics + name + '.csv', index=False)

        # --------------------------------------------------------------------------------------------------
        # STEP 5: Save model
        self.save_model()


    def predict(self, X, bias_type = None, y = None):
        """
        :param X: Predictors of test set.
        :param bias_type: Age bias selection. Options: 'cole', 'beheshti_1', 'beheshti_2', 'lange' or None (no age bias is applied). Default: None.
        :param y: Chronological age (required for 'beheshti_2' and 'lange'.
        :return: Apply developed model and selected biase over X to predict brain age.
        """
        '''
        X_scaled = self.scaler.transform(X[self.predictors])
        y_pred = self.model.predict(X_scaled)'''
        y_pred = self.model.predict(X[self.predictors])
        if bias_type == 'cole':
            return -self.beta['cole'] / self.alpha['cole'] + y_pred / self.alpha['cole']
        elif bias_type == 'beheshti_1':
            return (y_pred - self.beta['beheshti']) / (1 + self.alpha['beheshti'])
        elif bias_type == 'beheshti_2':
            return y_pred - self.alpha['beheshti'] * y - self.beta['beheshti']
        elif bias_type == 'lange':
            return y_pred + (1 - self.alpha['cole']) * y - self.beta['cole']
        elif bias_type == 'zhang':
            correction = age_level_correction.zhang(CA = y, BA = y_pred, age_limits=self.age_limits, zhang_parameters = self.zhang_parameters)
            correction.age_rounding()
            correction.age_level_bias_application()
            return correction.data['brain_age_corrected'].values
        else:
            return y_pred

    def save_model(self):
        """
        :return: Save developed model/coefficients in .pkl file in predefined directory.
        """
        #joblib.dump(self, self.directory_models + self.name + '.pkl')

        with open(self.directory_models + self.name + '.pkl', 'wb') as file:
            pickle.dump(self, file)

        self.examined_combinations.append(self.combination)
        if self.update_history_combinations:
            self.update_history()

    def update_history(self):
        """
        :return: Update associated .json file with already executed models train configuration/corrected models/tested models info
        """
        data = {'examined_combinations' : self.examined_combinations, 'corrected_ids' : self.corrected_ids,'id' : self.id + 1, 'tested' : self.tested}
        with open(self.historic_filepath, 'w') as file:
            json.dump(data, file)

    @classmethod
    def load_model(cls, model_name):
        """
        :param model_name: Model to be uploaded.
        :return: Upload model.
        """
        #model = joblib.load(model_name)
        with open(model_name, 'rb') as file:
            model = pickle.load(file)
        return model

