import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import nibabel as nib
from scipy.ndimage import zoom
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

import src.constants as C
from src.models import ResNet, BasicBlock, Bottleneck


def KLDivLoss(y_true, y_pred):
    """Returns K-L Divergence loss
    Different from the default TensorFlow KLDivergence in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    # Convert to tensors and ensure dtype is float32
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Prevent log(0) issues
    y_true = tf.math.maximum(y_true, 1e-16)

    # Calculate KL Divergence
    log_target = False
    if log_target:
        kl_div = tf.math.exp(y_true) * (y_true - y_pred)
    else:
        kl_div = y_true * (tf.math.log(y_true) - y_pred)
    loss = tf.reduce_mean(kl_div, axis=-1)

    return loss

class AWDLoss:
    def __init__(self, num_bins=C.BINS, initial_alpha=C.ALPHA, initial_beta=C.BETA, initial_w_coeffs=None):
        self.num_bins = num_bins
        self.alpha = tf.Variable(initial_alpha, trainable=False)
        self.beta = tf.Variable(initial_beta, trainable=False)
        if initial_w_coeffs is None:
            initial_w_coeffs = [1.0] * num_bins 
        self.w_coeffs = tf.Variable(initial_w_coeffs, dtype=tf.float32, trainable=False)
    def __call__(self, y_true, y_pred):
        errors = tf.abs(y_true - y_pred)
        overall_mae = tf.reduce_mean(errors)
        # Dynamically calculate age bins based on the number of bins and age range
        age_min = tf.reduce_min(y_true)
        age_max = tf.reduce_max(y_true)
        age_range = tf.maximum(age_max - age_min, 1e-5)
        bin_size = age_range / self.num_bins
        age_bins = tf.range(start=age_min, limit=age_max, delta=bin_size, dtype=tf.float32)
        # TensorArray to store MAEs for each bin
        maes = tf.TensorArray(dtype=tf.float32, size=self.num_bins)
        def calculate_bin_mae(i, maes):
            age_bin_start = age_bins[i]
            age_bin_end = age_bin_start + bin_size
            bin_indices = tf.where((y_true >= age_bin_start) & (y_true < age_bin_end))
            bin_errors = tf.gather_nd(errors, bin_indices)
            bin_mae = tf.cond(tf.size(bin_errors) > 0, lambda: tf.reduce_mean(bin_errors), lambda: tf.constant(0.0, dtype=tf.float32))
            return i + 1, maes.write(i, bin_mae)

        _, maes = tf.while_loop(lambda i, _: i < self.num_bins, calculate_bin_mae, [0, maes])
        maes = maes.stack()

        weighted_maes = tf.reduce_sum(self.w_coeffs * maes)
        loss = self.alpha * overall_mae + self.beta * weighted_maes
        return loss

class CustomMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='custom_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ages = tf.constant(np.arange(C.LOW_BIN, C.HGH_BIN - 1 + C.STEP_BIN, C.STEP_BIN), dtype=tf.float32)

    def calculate_representative_age_tf(self, distribution):
        # Ensure the ages are broadcastable with the distribution
        return tf.reduce_sum(self.ages * distribution, axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Cast inputs to the same dtype
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # Convert log probabilities to probabilities using softmax
        y_pred_prob = tf.nn.softmax(y_pred)
        
        # Compute expected ages
        y_true_age = tf.map_fn(self.calculate_representative_age_tf, y_true, dtype=tf.float32)
        y_pred_age = tf.map_fn(self.calculate_representative_age_tf, y_pred_prob, dtype=tf.float32)
        
        # Call the parent class's update_state method
        return super().update_state(y_true_age, y_pred_age, sample_weight)

class CustomSiMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='custom_mae', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Ensure y_true and y_pred are of type float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Call the parent class's update_state method with the true and predicted ages
        return super().update_state(y_true, y_pred, sample_weight)
    
class CustomMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, name='custom_mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ages = tf.constant(np.arange(C.LOW_BIN, C.HGH_BIN - 1 + C.STEP_BIN, C.STEP_BIN), dtype=tf.float32)

    def calculate_representative_age_tf(self, distribution):
        # Ensure the ages are broadcastable with the distribution
        return tf.reduce_sum(self.ages * distribution, axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert log probabilities to probabilities using softmax
        y_pred_prob = tf.nn.softmax(y_pred)
        
        # Compute expected ages
        y_true_age = tf.map_fn(self.calculate_representative_age_tf, y_true, dtype=tf.float32)
        y_pred_age = tf.map_fn(self.calculate_representative_age_tf, y_pred_prob, dtype=tf.float32)
        
        # Call the parent class's update_state method
        return super().update_state(y_true_age, y_pred_age, sample_weight)
    
class CustomSiMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, name='custom_mse', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assuming y_true and y_pred are directly the age values (e.g., float numbers)

        # Ensure y_true and y_pred are of type float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Call the parent class's update_state method with the true and predicted ages
        return super().update_state(y_true, y_pred, sample_weight)

    
def load_scan(scan_path, label):
    def _load_scan(scan_path):
        scan_path = scan_path.numpy().decode('utf-8')
        scan = nib.load(scan_path)
        scan_array = scan.get_fdata()
        target_shape = (C.CSIZE, C.CSIZE, C.CSIZE)
        zoom_factors = [n / o for n, o in zip(target_shape, scan_array.shape)]
        resized_scan_array = zoom(scan_array, zoom_factors, order=1)
        scan_tensor = tf.convert_to_tensor(resized_scan_array, dtype=tf.float32)
        scan_tensor = tf.expand_dims(scan_tensor, axis=-1)
        scan_tensor = scan_tensor / tf.reduce_max(scan_tensor)
        return scan_tensor

    scan_tensor = tf.py_function(_load_scan, [scan_path], Tout=tf.float32)
    scan_tensor.set_shape([C.CSIZE, C.CSIZE, C.CSIZE, 1])
    return tf.cast(scan_tensor, tf.float16), label

class BinMAELogger(tf.keras.callbacks.Callback):
    def __init__(self, num_bins, log_dir):
        super(BinMAELogger, self).__init__()
        self.num_bins = num_bins
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        bin_mae = self.model.evaluate(self.validation_data[0], self.validation_data[1], verbose=0)
        with open(os.path.join(self.log_dir, f'epoch_{epoch + 1}_bin_mae.txt'), 'w') as f:
            for i in range(self.num_bins):
                f.write(f'Bin {i + 1}: {bin_mae[i]}\n')
        print(f'\nEpoch {epoch + 1}: Bin MAE logged in {self.log_dir}/epoch_{epoch + 1}_bin_mae.txt')


def load_model(deep=True, layers=[3, 4, 6, 3], num_classes=40, channel_size=[64, 64, 128, 256, 512], dropout=True):
    """
    Load or define the model architecture.
    """
    if deep:
        class DeepResNet(ResNet):
            def __init__(self):
                super().__init__(Bottleneck, layers, num_classes, channel_size, dropout)
        model = DeepResNet()
    else:
        class ShallowResNet(ResNet):
            def __init__(self):
                super().__init__(BasicBlock, layers, num_classes, channel_size, dropout)
        model = ShallowResNet()
    return model

class MetricsLogger(Callback):
    def __init__(self, log_file_path):
        super(MetricsLogger, self).__init__()
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure the log directory exists

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"Epoch {epoch + 1}:\n")
            log_file.write(f" - loss: {logs.get('loss')}\n")
            log_file.write(f" - mean_absolute_error: {logs.get('custom_mae')}\n")
            log_file.write(f" - pearson: {logs.get('pearson_correlation')}\n")
            log_file.write(f" - val_loss: {logs.get('val_loss')}\n")
            log_file.write(f" - val_mean_absolute_error: {logs.get('val_custom_mae')}\n")
            log_file.write(f" - val_pearson: {logs.get('val_pearson_correlation')}\n")
            log_file.write("\n")

### Saving ###

def save_subject_ids(output_dir, fold_num, X_train, X_test):
    """
    Save the subject IDs for the training and test sets for a given fold.
    """
    os.makedirs(output_dir, exist_ok=True)
    fold_subjects = {
        'train': X_train.tolist(),
        'test': X_test.tolist()
    }
    subjects_json_path = os.path.join(output_dir, f'subject_ids_fold_{fold_num}.json')
    with open(subjects_json_path, 'w') as json_file:
        json.dump(fold_subjects, json_file, indent=4)

class SaveWeightsEveryNEpochs(Callback):
    def __init__(self, save_freq, save_dir):
        super(SaveWeightsEveryNEpochs, self).__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            filepath = os.path.join(self.save_dir, f"weights_epoch_{epoch + 1}.h5")
            self.model.save_weights(filepath)
            print(f"\nSaved weights at epoch {epoch + 1} to {filepath}")

def log_json(fold_num, checkpoint_path, X_train, X_test, history, output_file):
    print(fold_num)
    if C.NO_FOLDS==None:
        fold_num = 999
    # Convert NumPy arrays to lists
    X_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
    X_test_list = X_test.tolist() if isinstance(X_test, np.ndarray) else X_test
    
    fold_info = {
        "fold_number": fold_num,
        "best_model_weights_path": checkpoint_path,
        "training_set_ids": X_train_list,
        "test_set_ids": X_test_list,
        "epochs": []
    }

    # Collect available metrics from the history object
    for epoch in range(len(history.history['loss'])):
        epoch_info = {
            "epoch": epoch + 1
        }

        # Add all other metrics dynamically
        for metric in history.history.keys():
            if metric.startswith('val_'):
                epoch_info[f"val_{metric[4:]}"] = history.history[metric][epoch]
            else:
                epoch_info[f"{metric}"] = history.history[metric][epoch]

        # Convert arrays to lists for JSON serialization
        for key, value in epoch_info.items():
            if isinstance(value, np.ndarray):
                epoch_info[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                epoch_info[key] = value.item()

        fold_info["epochs"].append(epoch_info)

    # Initialize the folds_info dictionary
    folds_info = {"folds": []}

    # Load existing folds info if the output file exists and is not empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'r') as f:
                folds_info = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading existing JSON data from {output_file}: {e}")
    
    folds_info["folds"].append(fold_info)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(folds_info, f, indent=4)
    except TypeError as e:
        # Handle the error and save to a second file
        print(f"Error saving to {output_file}: {e}")

def calculate_representative_age(distribution):
    """
    Calculate the representative age from a probability distribution.

    Parameters:
        distribution (list or array-like): Probability distribution for ages.
        low_bin (float): Lower bound of the age range.
        high_bin (float): Upper bound of the age range.
        interval (float): Interval between discrete ages in the distribution.

    Returns:
        float: Representative age.
    """
    ages = np.arange(C.LOW_BIN, C.HGH_BIN, C.STEP_BIN)
    representative_age = np.sum(ages * distribution)
    return representative_age

def discretize_ages(ages, bin_size=C.STEP_BIN):
    """
    Discretize the ages into bins based on the min and max values with specified bin size.

    Parameters:
        ages (array-like): List or array of ages.
        bin_size (int): Size of each bin.

    Returns:
        array-like: Discretized age bins.
        list: List of bin edges.
    """
    # If ages are probability distributions, take their mean value
    if isinstance(ages[0], (np.ndarray, list)):
        ages = [calculate_representative_age(age) for age in ages]

    min_age = C.LOW_BIN
    max_age = C.HGH_BIN + 1.
    bins = np.arange(min_age, max_age + bin_size, bin_size)  

    # Discretize the ages using the bins
    age_bins = pd.cut(ages, bins=bins, labels=False, include_lowest=True)

    # Calculate the distribution in each bin
    bin_counts = np.bincount(age_bins)
    total_cases = len(ages)
    
    return age_bins, bins


### Age correction ###

def correct_age_predictions(true_ages, predicted_ages):
    """
    Corrects age predictions by regressing out the age from the brain-age delta.
    
    Parameters:
    true_ages (np.array): Array of true chronological ages.
    predicted_ages (np.array): Array of predicted ages by the model.
    
    Returns:
    corrected_predictions (np.array): Array of corrected predicted ages.
    original_corr (float): Spearman's rank correlation before correction.
    corrected_corr (float): Spearman's rank correlation after correction.
    """
    
    # Step 1: Calculate the brain-age delta
    brain_age_delta = predicted_ages - true_ages

    # Step 2: Perform linear regression
    X = true_ages.reshape(-1, 1)  # Reshape for sklearn which expects a 2D array
    y = brain_age_delta

    regressor = LinearRegression()
    regressor.fit(X, y)

    # Step 3: Adjust the delta
    fitted_values = regressor.predict(X)
    adjusted_delta = brain_age_delta - fitted_values

    # Step 4: Correct the predictions
    corrected_predictions = true_ages + adjusted_delta

    # Compute Spearman's rank correlation before and after correction
    original_corr, _ = spearmanr(true_ages, brain_age_delta)
    corrected_corr, _ = spearmanr(true_ages, adjusted_delta)

    print(f"Original Spearman's r: {original_corr}")
    print(f"Corrected Spearman's r: {corrected_corr}")

    return corrected_predictions, original_corr, corrected_corr

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 0 and epoch % C.EPOCHS_MOD == 0:
        return lr * C.EPOCHS_RED
    return lr

class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_correlation', **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.pearson_correlation = self.add_weight(name='pc', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_mean = K.mean(y_true)
        y_pred_mean = K.mean(y_pred)
        covariance = K.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        y_true_std = K.std(y_true)
        y_pred_std = K.std(y_pred)
        correlation = covariance / (y_true_std * y_pred_std + K.epsilon())

        self.pearson_correlation.assign_add(correlation)
        self.count.assign_add(1.0)

    def result(self):
        return self.pearson_correlation / self.count

    def reset_state(self):
        self.pearson_correlation.assign(0.0)
        self.count.assign(0.0)

### DATA AUGMENTATION ###

def random_shift(image):
    """
    Randomly shift the image by 0, 1, or 2 voxels along each axis.
    """
    shift_x = tf.random.uniform([], minval=-2, maxval=3, dtype=tf.int32)
    shift_y = tf.random.uniform([], minval=-2, maxval=3, dtype=tf.int32)
    shift_z = tf.random.uniform([], minval=-2, maxval=3, dtype=tf.int32)
    image = tf.roll(image, shift_x, axis=0)
    image = tf.roll(image, shift_y, axis=1)
    image = tf.roll(image, shift_z, axis=2)
    return image

def random_mirror(image):
    """
    Randomly mirror the image about the sagittal plane (axis 0) with a probability of 50%.
    """
    mirrored = tf.random.uniform([], 0, 1) > 0.5
    image = tf.cond(mirrored, lambda: tf.image.flip_left_right(image), lambda: image)
    return image

def augment(scan, label):
    """
    Apply random shift and mirror augmentation to the scan.
    """
    scan = random_shift(scan)
    scan = random_mirror(scan)
    return scan, label

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def compute_alpha_beta(y_train_true, y_train_pred, y_valid_true, y_valid_pred, new_folder_name):

    # Initialize dictionaries to store alpha and beta values
    train_alpha = {}
    train_beta = {}
    valid_alpha = {}
    valid_beta = {}

    # Fit the first model (cole) - train
    train_age_bias_cole = LinearRegression()
    train_age_bias_cole.fit(y_train_true.reshape(-1, 1), y_train_pred)
    train_alpha['cole'] = train_age_bias_cole.coef_[0]
    train_beta['cole'] = train_age_bias_cole.intercept_

    # Fit the first model (cole) - valid
    valid_age_bias_cole = LinearRegression()
    valid_age_bias_cole.fit(y_valid_true.reshape(-1, 1), y_valid_pred)
    valid_alpha['cole'] = valid_age_bias_cole.coef_[0]
    valid_beta['cole'] = valid_age_bias_cole.intercept_

    # Fit the second model (beheshti) - train
    train_age_bias_beheshti = LinearRegression()
    train_age_bias_beheshti.fit(y_train_true.reshape(-1, 1), y_train_pred - y_train_true)
    train_alpha['beheshti'] = train_age_bias_beheshti.coef_[0]
    train_beta['beheshti'] = train_age_bias_beheshti.intercept_

    # Fit the second model (beheshti) - valid
    valid_age_bias_beheshti = LinearRegression()
    valid_age_bias_beheshti.fit(y_valid_true.reshape(-1, 1), y_valid_pred - y_valid_true)
    valid_alpha['beheshti'] = valid_age_bias_beheshti.coef_[0]
    valid_beta['beheshti'] = valid_age_bias_beheshti.intercept_

    # Combine all data into one dictionary to save as JSON
    data_to_save = {
        "train_alpha": train_alpha,
        "train_beta": train_beta,
        "valid_alpha": valid_alpha,
        "valid_beta": valid_beta
    }

    # Save the dictionary to a JSON file
    json_file_path = os.path.join(new_folder_name, "alpha_beta_values.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

    # Display the results
    print(f"Alpha (cole): {train_alpha['cole']}, Beta (cole): {train_beta['cole']}")
    print(f"Alpha (beheshti): {train_alpha['beheshti']}, Beta (beheshti): {train_beta['beheshti']}")
    print(f"Alpha (cole): {valid_alpha['cole']}, Beta (cole): {valid_beta['cole']}")
    print(f"Alpha (beheshti): {valid_alpha['beheshti']}, Beta (beheshti): {valid_beta['beheshti']}")

    return train_alpha, train_beta, valid_alpha, valid_beta

def create_experiment_folder():
    folder_name = f"model_lr{C.LEARNING_RATE}_epoch{C.EPOCHS}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

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