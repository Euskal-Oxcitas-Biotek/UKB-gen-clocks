import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from scipy.stats import t,f
from scipy.linalg import pinv


class metrics():
    def __init__(self, model = None, X = None, y = None, y_pred = None):
        """
        :param model: Model name of function used as brain clock.
        :param X: Training set
        :param y: Target (age) in training set.
        :param y_pred: Estimated brain age.
        :param bins: Age bins to be used to compute performance metrics by age groups.
        """
        self.model = model
        self.bins = model.bins
        self.X = X
        self.Y = pd.DataFrame({
            'age_at_scan': y,
            'brain_age': y_pred
        })

    def individual_components(self):
        """
        :return: Compute performance metrics on age prediction on full set (MAE/MSE) and by age groups (MAE/mean PAD)
        """
        self.mse = mean_squared_error(self.Y['age_at_scan'], self.Y['brain_age'])
        self.mae = mean_absolute_error(self.Y['age_at_scan'], self.Y['brain_age'])
        self.mae_bins = {}
        self.meanPAD_bins = {}

        for i in range(len(self.bins) - 1):
            bin_test = self.Y[
                (self.Y['age_at_scan'] >= self.bins[i]) & (self.Y['age_at_scan'] <= self.bins[i + 1])]

            self.mae_bins[
                str(self.bins[i]) + '-' + str(self.bins[i + 1])] = mean_absolute_error(
                bin_test['age_at_scan'],
                bin_test['brain_age'])
            self.meanPAD_bins[
                str(self.bins[i]) + '-' + str(self.bins[i + 1])] = np.mean(bin_test['age_at_scan'] - bin_test['brain_age'])


    def compute_p_values(self):
        """
        :return: Compute p-values per coefficient of the linear model (t-test-based)
        """
        n = self.X.shape[0]
        p = self.X.shape[1]
        # Compute residuals
        residuals = self.Y['age_at_scan'] - self.Y['brain_age']
        # Compute residual standard error
        rss = np.sum(residuals ** 2)
        dof = n - p - 1  # degrees of freedom
        mse = rss / dof
        #se = np.sqrt(np.diag(np.linalg.inv(np.dot(self.X.T, self.X)))) * np.sqrt(mse)
        se = np.sqrt(np.diag(pinv(np.dot(self.X.T, self.X)))) * np.sqrt(mse)
        # Compute t-statistics and p-values
        t_statistics = self.model.model.coef_ / se
        self.p_values = (1 - t.cdf(np.abs(t_statistics), dof)) * 2  # two-tailed test