import numpy as np
import pandas as pd
import json
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from numpy.ma.extras import unique
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import json
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

warnings.filterwarnings("ignore")

class plots_individual_UKBB():
    def __init__(self, data_location = "UKBB_male_standardized_results_MC.csv",
                 directory_results = 'post_analysis_standardized/UKBB/Male/', type_of_approach = None, healthy_definition = 'orx',
                 binary_cases = ["CNvsNoCN"], age_bounded = False, analysis_mode = 'standard',showfliers=False, normalized_analysis = False,
                 linear_methods=['elastic_net', 'ols', 'lasso', 'ridge'], non_linear_methods = ['tpot', 'flaml', 'lgb_hyperopt', 'xgb_hyperopt'],
                 generalization_group = None, name_origin = None, gender = None):
        """
        :param data_location: Location of dataframe with metrics to be analyzed.
        :param directory_data: Directory where we want to save all generated metadata. Default: Create subfolder named '/data'.
        :param type_of_approach: Define if there is one specific "Age_bias_correction" that we want to analyze, if None, analyze them all
        :param healthy_definition: Either to use healthy 'orx' or 'cole'.
        :param binary_cases: Specify in a list which binary conditions we want to study.
        :param age_bounded: Boolean to specify if we want to use bounded metrics or not.
        :param analysis_mode: Specify type of analysis to perform, options: ['standard', 'linear_standard', 'cole_resample', 'generalization']
        :param showfliers: Boolean to show or not outliers in boxplots.
        :param normalized_analysis: Whether or not to include analysis normalizing per case (see STEP 3-4)
        :param linear_methods: Specify the names (in 'Description') of the linear models. In 'MC': ['elastic_net', 'ols', 'lasso', 'ridge']
        :param non_linear_methods: Specify the names (in 'Description') of the non-linear models. In 'MC': ['tpot', 'flaml', 'lgb_hyperopt', 'xgb_hyperopt']
        :param generalization_group : String only used when analysis_mode = 'generalization'. Options: ['database', 'manufacturer', 'machine', 'ethnicity', 'CN_type', 'education_years']
        :param name_origin: String with .csv origin; e.g., 'MC', 'SV', 'GA'. Used for file naming and performing 'used_predictors_size' stats test.
        :return: Data split for training and test AVIV models
        """
        #---------------------------------------------------------------------------------------------------------------
        # STEP 0: Feature initialization
        self.type_of_approach = type_of_approach
        self.analysis_mode = analysis_mode
        self.showfliers=showfliers
        self.type = ['']
        self.healthy_definition = healthy_definition
        self.binary_cases = binary_cases
        self.directory_results = directory_results
        self.normalized_analysis = normalized_analysis
        self.linear_methods = linear_methods
        self.non_linear_methods = non_linear_methods
        self.generalization_group = generalization_group
        self.name_origin = name_origin
        self.gender = gender

        if age_bounded:
            self.age_bounded = "bounded"
        else:
            self.age_bounded = "unbounded"
        self.full_data = pd.read_csv(data_location)

        if self.gender == 'M':
            self.full_data = self.full_data[self.full_data['Training_sex'] != 'female']
        elif self.gender == 'F':
            self.full_data = self.full_data[self.full_data['Training_sex'] != 'male']

        self.full_data['method_type'] = np.where(self.full_data['Description'].isin(self.linear_methods), 'linear',
                                     np.where(self.full_data['Description'].isin(self.non_linear_methods), 'non-linear', 'unknown'))
        #---------------------------------------------------------------------------------------------------------------
        # STEP 1: Settings preselection & extraction of available binary cases
        if self.analysis_mode in ['standard_linear']:
            self.full_data = self.full_data[self.full_data['method_type']=='linear']
        if self.type_of_approach is not None:
            self.full_data = self.full_data[self.full_data['Age_bias_correction'] == self.type_of_approach]

        self.available_binary_cases = []
        for binary_case in self.binary_cases:
            if not self.full_data['aurocBA_' + self.age_bounded + '_' + binary_case].isna().all():
                self.available_binary_cases.append(binary_case)
        print('In ' + data_location + '(' + str(
            self.type_of_approach) + ', ' + self.healthy_definition + ', ' + self.age_bounded + ') we have the following binary cases available:',
              str(self.available_binary_cases))
        #---------------------------------------------------------------------------------------------------------------
        # STEP 2: Compute max columns (MAE,meanPAD)
        self.full_data['max_MAE_bin_cole_unbounded'] = self.full_data[['MAE_cole_age.[55,60]','MAE_cole_age.[60,65]','MAE_cole_age.[65,70]','MAE_cole_age.[70,75]',
            'MAE_cole_age.[75,80]','MAE_cole_age.[80,85]']].max(axis=1)
        self.full_data['max_MAE_bin_orx_unbounded'] = self.full_data[['MAE_orx_age.[55,60]','MAE_orx_age.[60,65]','MAE_orx_age.[65,70]','MAE_orx_age.[70,75]',
            'MAE_orx_age.[75,80]','MAE_orx_age.[80,85]']].max(axis=1)
        self.full_data['max_abs_meanPAD_cole_unbounded'] = self.full_data[['meanPAD_cole_age.[55,60]','meanPAD_cole_age.[60,65]','meanPAD_cole_age.[65,70]','meanPAD_cole_age.[70,75]',
            'meanPAD_cole_age.[75,80]','meanPAD_cole_age.[80,85]']].abs().max(axis=1)
        self.full_data['max_abs_meanPAD_orx_unbounded'] = self.full_data[['meanPAD_orx_age.[55,60]','meanPAD_orx_age.[60,65]','meanPAD_orx_age.[65,70]','meanPAD_orx_age.[70,75]',
            'meanPAD_orx_age.[75,80]','meanPAD_orx_age.[80,85]']].abs().max(axis=1)
        self.full_data['mean_abs_meanPAD_cole_unbounded'] = self.full_data[['meanPAD_cole_age.[55,60]','meanPAD_cole_age.[60,65]','meanPAD_cole_age.[65,70]','meanPAD_cole_age.[70,75]',
            'meanPAD_cole_age.[75,80]','meanPAD_cole_age.[80,85]']].abs().mean(axis=1)
        self.full_data['mean_abs_meanPAD_orx_unbounded'] = self.full_data[['meanPAD_orx_age.[55,60]','meanPAD_orx_age.[60,65]','meanPAD_orx_age.[65,70]','meanPAD_orx_age.[70,75]',
            'meanPAD_orx_age.[75,80]','meanPAD_orx_age.[80,85]']].abs().mean(axis=1)

        self.full_data['max_MAE_bin_cole_bounded'] = self.full_data[['MAE_cole_age.[55,60]','MAE_cole_age.[60,65]','MAE_cole_age.[65,70]','MAE_cole_age.[70,75]',
            'MAE_cole_age.[75,80]']].max(axis=1)
        self.full_data['max_MAE_bin_orx_bounded'] = self.full_data[['MAE_orx_age.[55,60]','MAE_orx_age.[60,65]','MAE_orx_age.[65,70]','MAE_orx_age.[70,75]',
            'MAE_orx_age.[75,80]']].max(axis=1)
        self.full_data['max_abs_meanPAD_cole_bounded'] = self.full_data[['meanPAD_cole_age.[55,60]','meanPAD_cole_age.[60,65]','meanPAD_cole_age.[65,70]','meanPAD_cole_age.[70,75]',
            'meanPAD_cole_age.[75,80]']].abs().max(axis=1)
        self.full_data['max_abs_meanPAD_orx_bounded'] = self.full_data[['meanPAD_orx_age.[55,60]','meanPAD_orx_age.[60,65]','meanPAD_orx_age.[65,70]','meanPAD_orx_age.[70,75]',
            'meanPAD_orx_age.[75,80]']].abs().max(axis=1)
        self.full_data['mean_abs_meanPAD_cole_bounded'] = self.full_data[['meanPAD_cole_age.[55,60]','meanPAD_cole_age.[60,65]','meanPAD_cole_age.[65,70]','meanPAD_cole_age.[70,75]',
            'meanPAD_cole_age.[75,80]']].abs().mean(axis=1)
        self.full_data['mean_abs_meanPAD_orx_bounded'] = self.full_data[['meanPAD_orx_age.[55,60]','meanPAD_orx_age.[60,65]','meanPAD_orx_age.[65,70]','meanPAD_orx_age.[70,75]',
            'meanPAD_orx_age.[75,80]']].abs().mean(axis=1)
        self.full_data['1-abs(corr[PAD,CA]_' + self.healthy_definition + '_' + self.age_bounded + ')'] = 1 - abs(
            self.full_data['corr[PAD,CA]_' + self.healthy_definition + '_' + self.age_bounded])
        if self.analysis_mode in ['standard','standard_linear']:
            if self.normalized_analysis:
                self.type.append('normalized_')
                #---------------------------------------------------------------------------------------------------------------
                # STEP 3: Normalize MAE results -minimize- (in this case normalization is a relative error against the best-performing approach, the closest to zero the better)
                grouped = self.full_data.groupby(['Training_healthy_group', 'Training_sex', 'Oversampling', 'features_type']) #observe that when normalizing approaches we do it between those with the same training settings
                for col in ['MAE_orx_bounded','MAE_cole_bounded','MAE_orx_unbounded','MAE_cole_unbounded',
                            'max_MAE_bin_cole_bounded','max_MAE_bin_orx_bounded','max_abs_meanPAD_cole_bounded','max_abs_meanPAD_orx_bounded',
                            'max_MAE_bin_cole_unbounded','max_MAE_bin_orx_unbounded','max_abs_meanPAD_cole_unbounded','max_abs_meanPAD_orx_unbounded',
                            'mean_abs_meanPAD_cole_bounded','mean_abs_meanPAD_orx_bounded','mean_abs_meanPAD_cole_unbounded','mean_abs_meanPAD_orx_unbounded']:
                    self.full_data['min_'+col] = grouped[col].transform('min')
                    self.full_data['normalized_'+col] = 100*(self.full_data[col] - self.full_data['min_'+col]) / self.full_data[
                        'min_'+col]
                    self.full_data = self.full_data.drop(columns=['min_'+col])
                #---------------------------------------------------------------------------------------------------------------
                # STEP 4: Normalize AUROC and correlation results and determine binary cases available -maximimize-
                # (in this case normalization is ratio against the best_performing approach, the closest to one the better)
                for col in ['1-abs(corr[PAD,CA]_' + self.healthy_definition + '_' + self.age_bounded + ')',
                            'corr[BA,CA]_' + self.healthy_definition + '_' + self.age_bounded]:
                    self.full_data['max_'+col] = grouped[col].transform('max')
                    self.full_data['normalized_'+col] = 100*(self.full_data[col]) / self.full_data['max_'+col]
                    self.full_data = self.full_data.drop(columns=['max_' + col])

                for binary_case in self.available_binary_cases:
                    for feature in ['aurocBA_','aurocPAD_']:
                        col = feature + self.age_bounded + '_' + binary_case
                        self.full_data['max_' + col] = grouped[col].transform('max')
                        self.full_data['normalized_' + col] = 100 * (self.full_data[col]) / self.full_data['max_' + col]
                        self.full_data = self.full_data.drop(columns=['max_' + col])

        elif self.analysis_mode == 'cole_resample':
            #---------------------------------------------------------------------------------------------------------------
            # STEP 5: Prepare cases to compare models corrected via cole versus those trained with oversample
            self.full_data = self.full_data[
                ((self.full_data['Oversampling'] == 'default') & (self.full_data['Age_bias_correction'] == 'none')) |
                ((self.full_data['Age_bias_correction'] == 'cole') & (self.full_data['Oversampling'] == 'none')) |
                ((self.full_data['Age_bias_correction'] == 'none') & (self.full_data['Oversampling'] == 'none'))]
            self.full_data['method_type'] = np.nan
            # Update the 'Description' column based on the first condition: 'default' and 'none'
            self.full_data.loc[(self.full_data['Oversampling'] == 'default') & (
                        self.full_data['Age_bias_correction'] == 'none'), 'Description'] += '_resample'
            self.full_data.loc[(self.full_data['Oversampling'] == 'default') & (
                        self.full_data['Age_bias_correction'] == 'none'), 'method_type'] = 'resample'
            # Update the 'Description' column based on the second condition: 'cole' and 'none'
            self.full_data.loc[
                (self.full_data['Age_bias_correction'] == 'cole') & (self.full_data['Oversampling'] == 'none'), 'Description'] += '_cole'
            self.full_data.loc[
                (self.full_data['Age_bias_correction'] == 'cole') & (self.full_data['Oversampling'] == 'none'), 'method_type'] = 'cole'
            # Update the 'Description' column based on the second condition: 'none' and 'none'
            self.full_data.loc[
                (self.full_data['Age_bias_correction'] == 'none') & (self.full_data['Oversampling'] == 'none'), 'method_type'] = 'original'
        elif self.analysis_mode == 'generalization':
            sets_of_MAEs = {'database': ['db.ADNI','db.NACC','db.UKBB'], 'manufacturer':['manufacturer.GE','manufacturer.Philips','manufacturer.Siemens'],
                                 'machine': ['machine.Achieva','machine.Allegra','machine.Avanto','machine.Biograph.mMR','machine.DISCOVERY.MR750','machine.Espree','machine.GEMINI','machine.Gyroscan.NT',
                                              'machine.Ingenia','machine.Ingenuity','machine.Intera','machine.NUMARIS/4','machine.Prisma','machine.SIGNA','machine.Skyra','machine.Sonata','machine.Symphony',
                                              'machine.TrioTim','machine.Verio'],
                                 'ethnicity' : ['ethnicity.Asian','ethnicity.Black','ethnicity.Caribbean','ethnicity.Chinese','ethnicity.Mixed','ethnicity.NativeAmerican','ethnicity.Other','ethnicity.White'],
                                 'CN_type' : ['CN.0','CN.5','CN.10'], 'education_years' : ['education.[0,7]','education.[8,15]', 'education.[16,30]']}
            self.features_to_compare = ['MAE_'+ self.healthy_definition + '_' + self.age_bounded + '_' + feature for feature in sets_of_MAEs[self.generalization_group]]
        #---------------------------------------------------------------------------------------------------------------
        # STEP 6: Create directory where results will be stored
        self.results_root = self.directory_results.split('/')[0]+'/'#'post_analysis_standardized/'
        if not os.path.exists(self.directory_results):
            os.makedirs(self.directory_results, exist_ok=True)

    def distribution_boxplots(self, feature_set, subfolder = '/mae/'):
        """
        :param feature_set: Lists with features to generate the boxplots (this list can not be larger than of size 2)
        :param subfolder:  Subdirecory of self.directory_results + self.analysis_mode where we want to save the results
        :return: Distribution boxplots & associated stats tests.
        """
        if self.analysis_mode in ['standard','standard_linear']:
            #---------------------------------------------------------------------------------------------------------------
            # STEP 1: Create directory where results will be stored
            local_directory = self.directory_results + self.analysis_mode + subfolder
            if not os.path.exists(local_directory):
                os.makedirs(local_directory + 'stats/', exist_ok=True)
                os.makedirs(local_directory + 'plots/', exist_ok=True)
            #---------------------------------------------------------------------------------------------------------------
            # STEP 2: Generate multi-subplot figure and statistical tests per subplot
            for type in self.type: # Depending on the analysis_mode, perform analysis on row feature only or include normalized feature too
                plt.close()
                numeric_columns = [type+feature for feature in feature_set]
                #-----------------------------------------------------------------------------------------------------------
                #2.1. Palette selection
                base_colors = ['#1f77b4', '#ff69b4'] #original colors
                # Dynamically generate the rest of the color palette to match the length of 'feature_set'
                num_colors_needed = len(feature_set) - len(base_colors)  # Number of additional colors needed (if the user by mistake wants to show more than 2 metrics at the time)
                additional_colors = sns.color_palette("husl",
                                                      num_colors_needed)  # Using the 'husl' palette for visually distinct colors
                # Combine the base colors and the additional palette to create the full color list
                colors = base_colors + additional_colors.as_hex()
                #-----------------------------------------------------------------------------------------------------------
                fig, axes = plt.subplots(2, 3, figsize=(25, 10), constrained_layout=True)
                #-----------------------------------------------------------------------------------------------------------
                #2.1. Categorical features (6) -x-axis-
                categorical_columns = ['Description', 'Training_healthy_group', 'Training_sex', 'Oversampling', 'features_type']
                if self.type_of_approach is None:
                    categorical_columns.append('Age_bias_correction') #If type_of_approach is None; i.e., I do not se select am age-bias type. I compare between age-bias approaches
                else:
                    categorical_columns.append('method_type') #If type_of_approach is not None, I compare linear vs non-linear approaches
                # -----------------------------------------------------------------------------------------------------------
                #2.2. Generate plot
                for ax, cat_col in zip(axes.flatten(), categorical_columns):
                    if cat_col in self.full_data.columns:
                        unique_categories = np.arange(len(self.full_data[cat_col].unique()))
                        offset = 0.2
                        for idx,numeric_column in enumerate(numeric_columns):
                            sns.boxplot(data=self.full_data, x=cat_col, y=numeric_column, ax=ax, color=colors[idx],
                                        showfliers=self.showfliers, width=0.45,
                                        boxprops=dict(alpha=0.7),  # Adjust transparency to differentiate plots
                                        zorder=idx+1)  # Ensure proper layering of the plots

                        ax.set_xticks(unique_categories)
                        ax.set_xticklabels(self.full_data[cat_col].unique(), rotation=45)

                        ax.set_title(f"Distribution by {cat_col}")
                        ax.legend(handles=[
                            plt.Line2D([0], [0], color=color, lw=4) for color in colors
                        ], labels=numeric_columns, title="Numeric Columns")
                        #-------------------------------------------------------------------------------------------------------
                        # In cases where we may have numerical issues, use log-scale
                        if ('zhang' in self.full_data['Age_bias_correction'].values) and (self.analysis_mode != 'standard_linear'):
                            ax.set_yscale('log')
                        #-------------------------------------------------------------------------------------------------------
                        # 2.3. stat-tests per subplot
                        for numeric_column in numeric_columns:
                            self.stats_per_metric(self.full_data, cat_col, numeric_column, local_directory+ 'stats/', type)
                plt.savefig(local_directory + 'plots/' +  type + self.healthy_definition + '_' + str(self.type_of_approach) + "_" +self.age_bounded+".png")
                #plt.show()
            # 2.4. Perform analysis over the number if features used by linear models
            if self.analysis_mode == 'standard_linear' or self.name_origin == 'SV':
                self.stats_per_metric(self.full_data,'Description', 'used_predictors_set_size', local_directory+ 'stats/', '')
        else:
            print("distribution_boxplots() only available when analysis_mode in ['standard','standard_linear']")


    def generalization(self, model_feature):
        """
        :param model_feature: Model identification feature that we want to analyze for generabizability.
         Options = ['Description', 'Training_healthy_group', 'Training_sex', 'Oversampling', 'features_type', 'Age_bias_correction', 'method_type']
        :return: Distribution boxplots & associated stats tests for generalization attempt per model feature.
        """
        if self.analysis_mode in ['generalization']:
            # ---------------------------------------------------------------------------------------------------------------
            # STEP 1: Create directory where results will be stored
            local_directory = self.directory_results + self.generalization_group +'/' # self.directory_results + self.analysis_mode + subfolder
            if not os.path.exists(local_directory):
                os.makedirs(local_directory + 'stats/', exist_ok=True)
                os.makedirs(local_directory + 'plots/', exist_ok=True)
            # ---------------------------------------------------------------------------------------------------------------
            # STEP 2: Generate multi-subplot figure and statistical tests per subplot
            unique_cases = self.full_data[model_feature].unique()
            num_cases = len(unique_cases)
            fig, axes = plt.subplots(num_cases, 1, figsize=(len(self.features_to_compare) * 2, num_cases * 5), squeeze=False)
            for idx, case in enumerate(unique_cases):
                case_data = self.full_data[self.full_data[model_feature] == case]
                axes[idx, 0].boxplot([case_data[feature].dropna() for feature in self.features_to_compare],
                                     labels=self.features_to_compare)
                axes[idx, 0].set_title(f"Model Feature: {case}")
                axes[idx, 0].tick_params(axis='x', rotation=45)
                case_data = pd.melt(case_data, value_vars=self.features_to_compare, var_name=model_feature + '_' + case, value_name=self.generalization_group)
                case_data.dropna(inplace=True)
                self.stats_per_metric(case_data, model_feature + '_' + case, self.generalization_group, local_directory + 'stats/', '')
            plt.tight_layout()
            plt.savefig(local_directory + 'plots/' + self.healthy_definition + '_' + str(
                self.type_of_approach) + "_" + self.age_bounded + "_" + model_feature + "_" + self.generalization_group + ".png")
            #plt.show()
        else:
            print("generalization() only available when analysis_mode in ['generalization']")

    def cole_resample(self, group_feature):
        """
        :param group_feature: Feature to use as categorical (x-axis). Commonly used: 'method','method_type'
        :param method_order: Order in which features should
        :return: Distribution boxplots & associated stats tests  to compare cole-corrected approaches vs resampled uncorrected approaches
        """
        if self.analysis_mode in ['cole_resample']:
            self.directory_results_global_cole_resample = self.directory_results + self.analysis_mode + '/'
            if not os.path.exists(self.directory_results_global_cole_resample):
                os.makedirs(self.directory_results_global_cole_resample + 'stats/', exist_ok=True)
                os.makedirs(self.directory_results_global_cole_resample + 'plots/', exist_ok=True)

            #---------------------------------------------------------------------------------------------------------------
            # STEP 0: Define the metrics to plot and their corresponding titles
            metrics = ['MAE_'+self.healthy_definition+'_'+self.age_bounded, 'max_MAE_bin_'+self.healthy_definition+'_'+self.age_bounded,
                       'corr[BA,CA]_'+self.healthy_definition+'_'+self.age_bounded, '1-abs(corr[PAD,CA]_' + self.healthy_definition + '_' + self.age_bounded + ')',
                       'max_abs_meanPAD_'+self.healthy_definition+'_'+self.age_bounded]
            #---------------------------------------------------------------------------------------------------------------
            # STEP 1: Sort unique balues of 'group_feature' so that cases of interest are placed next to each other
            # 1.1. Get the unique values from the 'Description' column
            unique_groups = self.full_data[group_feature].unique()
            # 1.2. Sort by the substring before the first underscore and then by the full string
            method_order = sorted(unique_groups, key=lambda x: (x.split('_')[0], x))
            fig, axes = plt.subplots(len(metrics), 1, figsize=(1.5 * len(unique_groups), len(metrics) * 5), constrained_layout=True)
            #---------------------------------------------------------------------------------------------------------------
            # STEP 2: Generate figure
            for i, metric in enumerate(metrics):
                sns.boxplot(data=self.full_data, x= group_feature, y=metric, ax=axes[i], order=method_order, palette='Set2')
                axes[i].set_title(f"Boxplot of {metric} by Method")
                axes[i].set_xlabel(group_feature)
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                self.stats_per_metric(self.full_data, group_feature, metric, self.directory_results_global_cole_resample + 'stats/', '')
            plt.savefig(
                self.directory_results_global_cole_resample + 'plots/' + self.healthy_definition + "_" + self.age_bounded + "_"+ group_feature+".png")
            #plt.show()
        else:
            print("cole_resample() only available when analysis_mode in ['cole_resample']")

    def stats_per_metric(self, data, col, metric, local_directory, type):
        """
        :param data: Dataset to analyze (self.full_data or a subset of it)-
        :param col: categorical feature used to generate the comparison; e.g. col = 'Description'
        :param metric: numerical feature used to perform the statistical tests; e.g. metric = 'MAE_orx_unbounded'
        :param local_directory: directory where results will be saved
        :param type: Just for the naming of the file (either '' or 'normalized')
        :return:
        """
        data = data.dropna(subset=[col, metric])
        value_counts = data[col].value_counts()
        min_count = value_counts.min()
        num_tests = len(data[col].unique()) # count amount of groups to decide type of test to perform
        if (num_tests > 1) and (min_count >= 3): #min_count restriction added to be sure Shapiro's test can be ran
            #---------------------------------------------------------------------------------------------------------------
            # STEP 0: Tests for assumptions (group normality + equal variances)
            normality_p_values = []
            for origin in data[col].unique():
                group_data = data[data[col] == origin][metric]
                # 0.1. Perform Shapiro normality test on each group
                stat, p_value = stats.shapiro(group_data)
                normality_p_values.append(p_value)
            # -----------------------------------------------------------------------------
            # 0.2. Test for homogeneity of variances (Levene's test)
            stat, levene_p_value = stats.levene(
                *[data[data[col] == group][metric] for group in data[col].unique()])
            #---------------------------------------------------------------------------------------------------------------
            # STEP 1: Stat-test selection
            if  num_tests > 2: # If multi-group, decide between ANOVA and Kruskal-Wallis
                if all(p > 0.05 / num_tests for p in normality_p_values) and levene_p_value > 0.05: # ANOVA (normality and homogeneity condition satisfied)
                    stat_test_used = 'ANOVA'
                    data['metric'] = data[metric].copy()
                    data['col'] = data[col].copy()
                    model = ols('metric ~ C(col)', data=data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    p_value_multigroup = anova_table["PR(>F)"][0]
                    # 1.1. For further detail apply PostHoc (independently of ANOVA's p-value): Tukey HSD
                    tukey = pairwise_tukeyhsd(endog=data['metric'], groups=data['col'], alpha=0.05)
                    # 1.2. Save Tukey HSD results to a text file
                    with open(local_directory + "anova_tukey_" + type + self.healthy_definition + "_" + str(self.type_of_approach) + "_" + self.age_bounded + "_" + metric + "_" + col +".txt", 'w') as f:
                        f.write(f"ANOVA Results:\n{anova_table.to_string()}\n")
                        f.write("Tukey HSD Test Post-hoc Results:\n")
                        f.write(tukey.summary().as_text())
                else: # Kruskal-Wallis (normality and/or homogeneity condition not satisfied)
                    stat_test_used = 'Kruskal-Wallis'
                    stat, p_value_multigroup = stats.kruskal(
                        *[data[data[col] == group][metric] for group in data[col].unique()])
                    # 1.3. For further detail apply PostHoc (independently of Kruskall-Wallis' p-value): Tukey HSD
                    dunn_results = posthoc_dunn(data, val_col=metric, group_col=col, p_adjust='bonferroni')
                    # 1.4. Save Kruskal-Wallis and Dunn's results to a text file
                    with open(local_directory + "kruskall_dunn_" + type + self.healthy_definition + "_" + str(self.type_of_approach) + "_" + self.age_bounded + "_" + metric + "_" + col +".txt", 'w') as f:
                        f.write(f"Kruskal-Wallis Results:\nStatistic: {stat}, p-value: {p_value_multigroup}\n")
                        f.write("Dunn’s Test Post-hoc Results:\n")
                        f.write(dunn_results.to_string())

            else: # If dual-group, decide between t-test, Welch's and Mann-Whitney U-test (similar as before, this should be normalised)
                group1, group2 = [data[data[col] == arm][metric] for arm in
                                  data[col].unique()]
                # 1.5. Selection of test
                if all(p > 0.05 / num_tests for p in normality_p_values) and levene_p_value > 0.05:
                    # 1.5.1. If normality and equal variances are satisfied
                    test = ttest_ind(group1, group2)
                    stat_test_used = "Independent t-test"
                elif all(p > 0.05 / num_tests for p in normality_p_values):
                    # 1.5.2. If normality is satisfied but variances are not equal
                    test = ttest_ind(group1, group2, equal_var=False)
                    stat_test_used = "Welch's t-test"
                else:
                    # 1.5.3. If normality is not satisfied
                    test = mannwhitneyu(group1, group2)
                    stat_test_used = "Mann-Whitney U-test"
                p_value_multigroup = test.pvalue
            #---------------------------------------------------------------------------------------------------------------
            # STEP 2: If we have at least two groups, compute confidence intervals and save info in dataframe
            # 2.1. Function to calculate confidence interval

            def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
                mean = np.mean(data)# Original mean
                means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)] # Generate bootstrap samples and compute means
                lower_bound = np.percentile(means, (1 - confidence) / 2 * 100)# Compute the lower and upper percentiles
                upper_bound = np.percentile(means, (1 + confidence) / 2 * 100)
                margin_of_error = upper_bound - mean# Margin of error is the difference between the mean and the CI bounds
                return (mean, margin_of_error, lower_bound, upper_bound)

            ci_dict = {}
            for group, group_data in data.groupby(col):
                ci_dict[group] = bootstrap_ci(group_data[metric])#compute_ci(group_data[metric])

            cases_count = data[col].value_counts().to_dict()
            df = pd.DataFrame([[self.directory_results, self.analysis_mode, self.healthy_definition, str(self.type_of_approach), self.age_bounded, col, metric,
                                cases_count,stat_test_used,p_value_multigroup,str(ci_dict)]],
                              columns=['directory_results','analysis_mode','healthy_definition', 'type_of_approach', 'age_bounded', 'group_feature', 'metric_feature',
                                       'cases_count','stat_test','p_value','confidence_intervals'])
            df.to_csv(local_directory + self.healthy_definition + '_' + str(self.type_of_approach) + "_" +self.age_bounded+"_"+ metric+ "_" + col +".csv", index=False)


    def collector(self):
        """
        :return: Collect all .csv files (with the stats tests info) from the root directory and any sub-directories of it and put it in a single dataframe
        """
        df_list = []
        for root, dirs, files in os.walk(self.results_root):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    df_list.append(df)
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(self.results_root+'combined_data_tests_'+self.name_origin+'.csv', index=False)
