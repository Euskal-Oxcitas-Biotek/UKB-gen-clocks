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

    # Print values for debugging
    # tf.print("y_true:", y_true, summarize=-1)
    # tf.print("y_pred (log-prob):", y_pred, summarize=-1)
    # tf.print("y_pred (prob):", y_pred_log_prob, summarize=-1)

    return loss

# def composite_loss(y_true, y_pred):

#     kl_loss = KLDivLoss()(y_true, y_pred)
#     mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

#     return kl_loss + mae_loss
    
    # loss = tf.keras.losses.KLDivergence(reduction = tf.keras.losses.Reduction.SUM)(y_true, y_pred)
    # n = tf.cast(tf.shape(y_true)[0], tf.float32)
    
    return loss / n

# def AWDLoss(y_true, y_pred):
#     """ 
#     Returns the Adaptive Weighted Dynamic (AWD) Loss function.
    
#     This loss function incorporates overall Mean Absolute Error (MAE)
#     and MAE for each age bin, with adaptive weighting coefficients.
#     """
    
#     # Calculate absolute errors
#     errors = tf.abs(y_true - y_pred)

#     # Define age bins (every 10 years)
#     age_bins = [(i, i+10) for i in range(0, 100, 10)]

#     # Calculate overall MAE
#     overall_mae = tf.reduce_mean(errors)

#     # Calculate MAE for each age bin
#     maes = []
#     for age_bin in age_bins:
#         bin_indices = tf.where(tf.logical_and(y_true >= age_bin[0], y_true < age_bin[1]))
#         bin_errors = tf.gather(errors, bin_indices)
#         bin_mae = tf.reduce_mean(bin_errors)
#         maes.append(bin_mae)

#     # Define initial values of weighting coefficients (can be trainable variables)
#     alpha = tf.Variable(1.0, trainable=True)
#     beta = tf.Variable(1.0, trainable=True)
#     w_coeffs = [tf.Variable(1.0, trainable=True) for _ in range(len(age_bins))]

#     # Combine overall MAE and weighted sum of MAE_j
#     loss = alpha * overall_mae + beta * tf.reduce_sum([w * mae for w, mae in zip(w_coeffs, maes)])
    
#     # Optionally update weighting coefficients using gradient descent
#     gradient = False
#     if gradient:
#         with tf.GradientTape() as tape:
#             grads = tape.gradient(loss, [alpha, beta] + w_coeffs)
        
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Adjust learning rate as needed
#         optimizer.apply_gradients(zip(grads, [alpha, beta] + w_coeffs))

#     return loss

class AWDLoss:
    def __init__(self, num_bins=C.BINS, initial_alpha=C.ALPHA, initial_beta=C.BETA, initial_w_coeffs=None):
        self.num_bins = num_bins
        self.alpha = tf.Variable(initial_alpha, trainable=False)
        self.beta = tf.Variable(initial_beta, trainable=False)
        if initial_w_coeffs is None:
            initial_w_coeffs = [1.0] * num_bins  # Ensure this matches `num_bins`
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
        # Use tf.print for debugging information
        # tf.print("age_min:", age_min, "age_max:", age_max, "bin_size:", bin_size, "age_bins:", age_bins)
        _, maes = tf.while_loop(lambda i, _: i < self.num_bins, calculate_bin_mae, [0, maes])
        maes = maes.stack()
        # if len(self.w_coeffs) != len(maes):
        #     raise ValueError(f"Weight coefficients ({len(self.w_coeffs)}) do not match number of MAE bins ({len(maes)})")
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

        # Debugging: Print shapes and values
        # tf.print("y_true shape:", tf.shape(y_true))
        # tf.print("y_pred_prob shape:", tf.shape(y_pred_prob))
        # tf.print("self.ages shape:", tf.shape(self.ages))
        
        # Compute expected ages
        y_true_age = tf.map_fn(self.calculate_representative_age_tf, y_true, dtype=tf.float32)
        y_pred_age = tf.map_fn(self.calculate_representative_age_tf, y_pred_prob, dtype=tf.float32)
        
        # Debugging: Print expected ages
        # tf.print("y_true_age:", y_true_age)
        # tf.print("y_pred_age:", y_pred_age)
        
        # Call the parent class's update_state method
        return super().update_state(y_true_age, y_pred_age, sample_weight)

class CustomSiMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='custom_mae', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assuming y_true and y_pred are directly the age values (e.g., float numbers)

        # Ensure y_true and y_pred are of type float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Debugging: Print shapes and values
        # tf.print("y_true:", y_true)
        # tf.print("y_pred:", y_pred)
        
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

        # Debugging: Print shapes and values
        # tf.print("y_true shape:", tf.shape(y_true))
        # tf.print("y_pred_prob shape:", tf.shape(y_pred_prob))
        # tf.print("self.ages shape:", tf.shape(self.ages))
        
        # Compute expected ages
        y_true_age = tf.map_fn(self.calculate_representative_age_tf, y_true, dtype=tf.float32)
        y_pred_age = tf.map_fn(self.calculate_representative_age_tf, y_pred_prob, dtype=tf.float32)
        
        # Debugging: Print expected ages
        # tf.print("y_true_age:", y_true_age)
        # tf.print("y_pred_age:", y_pred_age)
        
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

        # Debugging: Print shapes and values
        # tf.print("y_true:", y_true)
        # tf.print("y_pred:", y_pred)
        
        # Call the parent class's update_state method with the true and predicted ages
        return super().update_state(y_true, y_pred, sample_weight)


class LocalMAE(tf.keras.metrics.Metric):
    def __init__(self, num_bins=C.BINS, name="local_mae", **kwargs):
        super(LocalMAE, self).__init__(name=name, **kwargs)
        self.num_bins = num_bins
        self.total_mae = self.add_weight(name="total_mae", initializer="zeros")
        self.bin_mae = self.add_weight(name="bin_mae", shape=(num_bins,), initializer="zeros")
        self.bin_counts = self.add_weight(name="bin_counts", shape=(num_bins,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Cast inputs to the same dtype
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        # Compute absolute errors
        errors = tf.abs(y_true - y_pred)
        
        # Dynamically calculate age bins based on the number of bins and age range
        age_min = tf.reduce_min(y_true)
        age_max = tf.reduce_max(y_true)
        age_range = tf.maximum(age_max - age_min, 1e-5)
        bin_size = age_range / self.num_bins
        age_bins = tf.range(start=age_min, limit=age_max, delta=bin_size, dtype=tf.float32)

        # Ensure age_bins has at least num_bins elements
        if tf.size(age_bins) < self.num_bins:
            age_bins = tf.concat([age_bins, [age_max]], axis=0)

        # Iterate over each bin and compute MAE for that bin
        for i in range(self.num_bins):
            age_bin_start = age_bins[i]
            age_bin_end = age_bin_start + bin_size
            bin_indices = tf.where((y_true >= age_bin_start) & (y_true < age_bin_end))
            bin_errors = tf.gather_nd(errors, bin_indices)
            bin_mae = tf.reduce_sum(bin_errors)
            bin_count = tf.cast(tf.size(bin_errors), tf.float32)

            # Update bin MAE and bin counts using TensorFlow operations
            indices = tf.convert_to_tensor([[i]], dtype=tf.int32)
            self.bin_mae.assign(tf.tensor_scatter_nd_add(self.bin_mae, indices, tf.convert_to_tensor([bin_mae], dtype=tf.float32)))
            self.bin_counts.assign(tf.tensor_scatter_nd_add(self.bin_counts, indices, tf.convert_to_tensor([bin_count], dtype=tf.float32)))
        
        # Update total MAE
        self.total_mae.assign_add(tf.reduce_sum(errors))

    def result(self):
        # Calculate the average MAE for each bin, handling bins with zero counts
        bin_mae_result = tf.where(self.bin_counts > 0, self.bin_mae / self.bin_counts, 0.0)
        return bin_mae_result

    def reset_state(self):
        # Reset the states for new evaluation
        self.total_mae.assign(0.0)
        self.bin_mae.assign(tf.zeros_like(self.bin_mae))
        self.bin_counts.assign(tf.zeros_like(self.bin_counts))

    
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
            # "train_loss": history.history['loss'][epoch],
            # "test_loss": history.history['val_loss'][epoch]
        }

        # Add all other metrics dynamically
        for metric in history.history.keys():
            # print(metric)
            # print(history.history[metric])
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
    # print(f"Min age: {min_age}")
    # print(f"Max age: {max_age}")
    # print(f"Bin step: {bin_size}")
    bins = np.arange(min_age, max_age + bin_size, bin_size)  # Create bins with float values

    # Discretize the ages using the bins
    age_bins = pd.cut(ages, bins=bins, labels=False, include_lowest=True)

    # Calculate the distribution in each bin
    bin_counts = np.bincount(age_bins)
    total_cases = len(ages)

    # print("Bin edges:", bins)
    # print("Distribution of cases in each bin:")
    # for i, count in enumerate(bin_counts):
    #     percentage = (count / total_cases) * 100
    #     print(f"Bin {i} ({bins[i]} - {bins[i+1]}): {count} cases ({percentage:.2f}%)")
    
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
    # print(f"Shifts: x={shift_x}, y={shift_y}, z={shift_z}")
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


class SaveAugmentationCallback(Callback):
    def __init__(self, model, train_batches, save_dir):
        super().__init__()
        self.model = model
        self.train_batches = iter(train_batches)  # Create an iterator from the dataset
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        # Fetch one batch from the dataset
        original_scan, label = next(self.train_batches)
        augmented_scan, _ = augment(original_scan[0], label)

        # Print shapes of original and augmented scans
        # print(f"Original scan shape: {original_scan[0].shape}")
        # print(f"Augmented scan shape: {augmented_scan.shape}")

        # Extract middle slices and squeeze to remove single dimension
        original_axial = np.squeeze(original_scan[0].numpy()[:, :, original_scan.shape[2] // 2])
        original_coronal = np.squeeze(original_scan[0].numpy()[:, original_scan.shape[1] // 2, :])
        original_sagittal = np.squeeze(original_scan[0].numpy()[original_scan.shape[0] // 2, :, :])

        augmented_axial = np.squeeze(augmented_scan.numpy()[:, :, augmented_scan.shape[2] // 2])
        augmented_coronal = np.squeeze(augmented_scan.numpy()[:, augmented_scan.shape[1] // 2, :])
        augmented_sagittal = np.squeeze(augmented_scan.numpy()[augmented_scan.shape[0] // 2, :, :])

        # Print shapes of extracted slices
        # print(f"Shapes - Original: axial {original_axial.shape}, coronal {original_coronal.shape}, sagittal {original_sagittal.shape}")
        # print(f"Shapes - Augmented: axial {augmented_axial.shape}, coronal {augmented_coronal.shape}, sagittal {augmented_sagittal.shape}")

        # Check if augmented_sagittal is empty
        # print(f"Augmented sagittal slice values: {augmented_sagittal}")

        # Create composite images for original and augmented scans
        original_composite = np.hstack((original_axial, original_coronal, original_sagittal))
        augmented_composite = np.hstack((augmented_axial, augmented_coronal, augmented_sagittal))

        # Stack original and augmented composites vertically
        combined_composite = np.vstack((original_composite, augmented_composite))

        # Create a larger canvas to include text annotations
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(combined_composite, cmap='gray')
        ax.axis('off')

        # Add text annotations
        ax.text(10, 10, 'Original', color='white', fontsize=12, weight='bold', backgroundcolor='black')
        ax.text(10, original_composite.shape[0] + 10, 'Augmented', color='white', fontsize=12, weight='bold', backgroundcolor='black')
        
        # Add plane annotations
        slice_width = original_axial.shape[1]
        ax.text(slice_width / 2, original_composite.shape[0] - 10, 'Axial', color='white', fontsize=12, weight='bold', ha='center', backgroundcolor='black')
        ax.text(slice_width + slice_width / 2, original_composite.shape[0] - 10, 'Coronal', color='white', fontsize=12, weight='bold', ha='center', backgroundcolor='black')
        ax.text(2 * slice_width + slice_width / 2, original_composite.shape[0] - 10, 'Sagittal', color='white', fontsize=12, weight='bold', ha='center', backgroundcolor='black')

        # Save the composite image with annotations
        composite_path = os.path.join(self.save_dir, f'epoch_{epoch+1}_composite.png')
        plt.savefig(composite_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # print(f"Epoch {epoch+1}: Saved composite image of original and augmented slices with annotations")


def visualize_augmentations(original_scan, augmented_scan, save_path):
    """
    Visualize the original and augmented MRI scans and save the visualization as an image file.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display original scan
    axes[0, 0].imshow(original_scan[original_scan.shape[0] // 2, :, :], cmap='gray')
    axes[0, 0].set_title('Original (Axial)')
    axes[0, 1].imshow(original_scan[:, original_scan.shape[1] // 2, :], cmap='gray')
    axes[0, 1].set_title('Original (Coronal)')
    axes[0, 2].imshow(original_scan[:, :, original_scan.shape[2] // 2], cmap='gray')
    axes[0, 2].set_title('Original (Sagittal)')
    
    # Display augmented scan
    axes[1, 0].imshow(augmented_scan[augmented_scan.shape[0] // 2, :, :], cmap='gray')
    axes[1, 0].set_title('Augmented (Axial)')
    axes[1, 1].imshow(augmented_scan[:, augmented_scan.shape[1] // 2, :], cmap='gray')
    axes[1, 1].set_title('Augmented (Coronal)')
    axes[1, 2].imshow(augmented_scan[:, :, augmented_scan.shape[2] // 2], cmap='gray')
    axes[1, 2].set_title('Augmented (Sagittal)')
    
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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