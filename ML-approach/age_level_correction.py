import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

class zhang():
    def __init__(self, CA, BA, rounding_strategy = 'down', age_limits = [55,85], std_threshold = 0,
                 window_age_level = 0, minimum_amount_warning = 10, zhang_parameters = None):
        """
        :param CA / BA: 'age_at_scan' or brain age predicted by a given model.
        :param rounding_strategy: Options to round "age_at_scan". Options -> 'down' (keep the integer of the age of the individual,
               i.e., 71.2 and 71.9 are both 71), 'nearest' (round to nearest integer, 71.2 would be 71, but 71.9 would be 72).
        :param age_limits: Age interval in years (list: [a,b]) denoting the age bracket considered during training/testing of developed models. Default: a=55,b=85.
        :param minimum_amount_warning: minimum amount of cases per rounded age (in training set), below which age user is warned the amount of cases
               available is below.
        :param window_age_level: Time window, in years, to be used to estimate mean and standard deviation of the age level correction.
        :param zhang_parameters: Dictionary with means and std for correction as described in Zhang.
        """
        self.rounding_strategy = rounding_strategy
        self.age_limits = age_limits
        self.window = window_age_level
        self.std_threshold = std_threshold
        self.minimum_amount_warning = minimum_amount_warning
        self.zhang_parameters = zhang_parameters
        self.data = pd.DataFrame({
            'brain_age': BA,
            'age_at_scan': CA,
            'PAD': BA-CA
        }).reset_index(drop=True)
        #print('data:',self.data)


    def age_rounding(self):
        """
        :return: Function to determine 'round_age' (self.RA), with the age rounding strategy selected.
        """
        if self.rounding_strategy == 'down':
            self.data['round_age']  = np.floor(self.data['age_at_scan']).astype(int)
        elif self.rounding_strategy == 'nearest':
            self.data['round_age'] = np.round(self.data['age_at_scan']).astype(int)

    def age_level_bias_training(self):
        """
        :return: Compute means and stds if rounding_strategy is not None. Save parameters in .json file.
        """
        #---------------------------------------------------------------------------------------------------------------
        # Step 0: Apply age rounding strategy on train and test sets
        self.age_rounding()
        #---------------------------------------------------------------------------------------------------------------
        # Step 1: If rounding strategy is not None, save means and stds for each age available in training set
        results = {}
        for key in range(self.age_limits[0],self.age_limits[1]+1):#sorted(self.train_data['round_age'].unique()):
            current_window = self.window - 1
            std_diff = 0
            while (std_diff == 0):
                current_window += 1
                subset = self.data[(self.data['round_age'] >= key-current_window) & (self.data['round_age'] <= key+current_window)] #overlapin window
                if len(subset) > self.minimum_amount_warning: #>0
                    differences = subset['PAD']
                    mean_diff = differences.mean()
                    std_diff = differences.std()
            results[key] = {'mean': mean_diff, 'std': std_diff, 'window': current_window, 'subset': len(subset)}
        self.zhang_parameters = results

    def age_level_bias_application(self):
        """
        :return: Apply age_level bias over test set
        """
        #---------------------------------------------------------------------------------------------------------------
        # Step 1: Apply age-level correction (PAD)
        self.data['PAD_c'] = None
        for idx, row in self.data.iterrows():
            round_age = row['round_age']
            mean = self.zhang_parameters[round_age]['mean']
            std = self.zhang_parameters[round_age]['std']
            self.data.at[idx, 'PAD_corrected'] = (row['PAD'] - mean) / std
        #---------------------------------------------------------------------------------------------------------------
        # Step 2: Correct brain age
        self.data['brain_age_corrected'] = self.data['PAD_corrected'] + self.data['age_at_scan']
