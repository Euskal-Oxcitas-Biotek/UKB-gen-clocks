import numpy as np
import tensorflow as tf
from scipy.stats import norm
from typing import Tuple, Union, List
import re

import sys

import json
from pathlib import Path

import src.constants as C

def load_data():

    pass

### Real Data ###

class Dataset:
    """
    A class to represent a dataset, allowing for the loading of dataset configurations
    and annotations.
    Attributes:
        name (str): The name of the dataset.
        scanIDs (list): A list of scan IDs included in the dataset.
        scans_path (str): Path to the directory where scan files are stored.
        targets_path (str): Path to the file containing annotations or targets.
    """
    
    def __init__(self, name: str, scanIDs: list, scans_path: str, targets_path: str):
        """
        Initializes the Dataset object with basic information.
        Parameters:
            name (str): The name of the dataset.
            scanIDs (list): List of unique identifiers for scans in the dataset.
            scans_path (str): Filesystem path to the scans associated with the dataset.
            targets_path (str): Filesystem path to the annotations or targets data.
        """
        self.name = name
        self.scanIDs = scanIDs
        print(len(self.scanIDs))
        self.scans_path = scans_path
        self.targets_path = targets_path
        self.ages = self.load_ages()

    def load_ages(self):
        """
        Loads ages from the JSON file specified in targets_path. This method includes checks
        to ensure the file exists and is a valid JSON file, raising appropriate errors if not.
        Returns:
            dict: A dictionary where each key is a scan ID and the value is the corresponding age.
        Raises:
            FileNotFoundError: If the JSON file does not exist at the specified path.
            ValueError: If the file is not a valid JSON file.
        """
        # Check if the JSON file exists
        targets_file = Path(self.targets_path)
        print("Target path: ", targets_file)
        if not targets_file.is_file():
            raise FileNotFoundError(f"Ages file {self.targets_path} not found.")
        
        # Attempt to load the JSON file, catching JSON decoding errors
        try:
            with open(targets_file, 'r') as file:
                ages = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"The file {self.targets_path} is not a valid JSON file.")

        return ages

    @staticmethod
    def from_file(file_name: Path) -> 'Dataset':
        """
        Static method to create a Dataset instance from a configuration file.
        Parameters:
            file_name (Path): Path object pointing to the JSON configuration file.
        Returns:
            Dataset: An instance of the Dataset class initialized with the data loaded from the file.
        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
        """
        if not file_name.is_file():
            raise FileNotFoundError(f"File {file_name} not found.")
        with open(file_name, 'r') as file:
            data = json.load(file)
            
        return Dataset(data["name"], data["scanIDs"], data["scans_path"], data["targets"])
        
    def train_test_iterator(self, k=C.NO_FOLDS, seed=42):
        """
        Generates train-test splits for k-fold cross-validation.
        Parameters:
            k (int): Number of folds for cross-validation.
            seed (int): Seed for the random number generator to ensure reproducibility.
        Yields:
            tuple: Four arrays containing the paths to the scan files for training, 
                   the paths to the scan files for testing, the age labels for training,
                   and the age labels for testing, for each fold.
        """
        random_arr = np.arange(len(self.scanIDs))
        np.random.seed(seed=seed)
        np.random.shuffle(random_arr)

        scan_files = []
        age_labels = []
        for idx in random_arr:
            scan_id = self.scanIDs[idx]
            scan_file = (Path(self.scans_path) / scan_id).as_posix()
            scan_files.append(scan_file)
        
            # Retrieve the age for the current scan ID
            age_data = self.ages.get(scan_id[:-4], None)
            if age_data is not None:
                age_labels.append(age_data)
            else:
                age_labels.append([-1])
                print("NONE in case ", scan_id)
        # print(age_labels)
        fold_sizes = np.repeat(len(self.scanIDs) // k, k)
        fold_sizes[:len(self.scanIDs) % k] += 1  # Adjust for non-divisible lengths

        num_fold = np.repeat(np.arange(k), fold_sizes)
        scan_files = np.array(scan_files)
        age_labels = np.array(age_labels)

        for fold in range(k):
            x_train = scan_files[num_fold != fold]
            x_test = scan_files[num_fold == fold]
            y_train = age_labels[num_fold != fold]
            y_test = age_labels[num_fold == fold]
            
            yield x_train, x_test, y_train, y_test

    @staticmethod
    def extract_subj_id(filename):
        # First pattern
        match = re.search(r'dADNI_(s\d+)_', filename)
        if match:
            return match.group(1)
        
        # Second pattern
        match = re.search(r'dUKBB_(s\d+)_', filename)
        if match:
            return match.group(1)
        
        # Third pattern
        match = re.search(r'dNACC_(s\d+)_', filename)
        if match:
            return match.group(1)
        
        return None
    
    @staticmethod
    def format_labels(labels):
        """
        Ensure the labels have a consistent format for NumPy arrays.
        """
        if isinstance(labels[0], dict) and 'age' in labels[0]:
            return np.array([item['age'] for item in labels])
        return np.array(labels)

    def train_test_subjID_iterator(self, k=C.NO_FOLDS, test_size=C.TEST_SIZE, seed=42):
        
        np.random.seed(seed=seed)

        # Initialize a dictionary to store data by subject
        subj_to_data = {}

        # Gather data by subject ID
        for scan_id in self.scanIDs:
            subj_id = self.extract_subj_id(scan_id)
            if subj_id not in subj_to_data:
                subj_to_data[subj_id] = {'scans': [], 'ages': []}
            
            scan_file = (Path(self.scans_path) / scan_id).as_posix()
            age_data = self.ages.get(scan_id[:-4], [-1])  # Assuming ages are mapped by scan_id without file extension
            subj_to_data[subj_id]['scans'].append(scan_file)
            subj_to_data[subj_id]['ages'].append(age_data)

        # Print the data collected for debugging
        # print(f"Collected data by subject: {subj_to_data}")  # Added for debugging

        # Shuffle the subject IDs
        subjects = list(subj_to_data.keys())
        # np.random.shuffle(subjects)

        if k is not None:
            # k-fold cross-validation
            fold_sizes = np.repeat(len(subjects) // k, k)
            fold_sizes[:len(subjects) % k] += 1  # Adjust for non-divisible lengths
            folds = np.split(subjects, np.cumsum(fold_sizes)[:-1])

            # Yield data for each fold
            for fold_idx in range(k):
                test_ids = set(folds[fold_idx])
                train_ids = set(subjects) - test_ids

                # Ensure there is no intersection between train and test sets
                assert train_ids.isdisjoint(test_ids), "Data leakage detected: Subject IDs are overlapping."

                x_train, y_train, x_test, y_test = [], [], [], []

                for subj_id in subjects:
                    data = subj_to_data[subj_id]
                    if subj_id in test_ids:
                        x_test.extend(data['scans'])
                        y_test.extend(data['ages'])
                    else:
                        x_train.extend(data['scans'])
                        y_train.extend(data['ages'])

                # Print the shapes and types of data for debugging
                print(f"Fold {fold_idx + 1} - Train Scans: {len(x_train)}, Test Scans: {len(x_test)}")  # Added for debugging

                # Convert labels to consistent format
                y_train = self.format_labels(y_train)  
                y_test = self.format_labels(y_test)    

                # Print formatted labels for debugging
                print(f"Formatted Train Labels Shape: {y_train.shape}, Test Labels Shape: {y_test.shape}")  # Added for debugging

                yield np.array(x_train), np.array(x_test), y_train, y_test

        else:
            # Simple train-test split
            test_count = int(len(subjects) * test_size)     # Calculate number of test subjects
            train_count = len(subjects) - test_count        # Calculate number of train subjects
            print(f"Total subjects: {len(subjects)}")
            print(f"Training subjects: {train_count}, Test subjects: {test_count}")

            # Split the subjects into training and test sets
            train_ids = set(subjects[:train_count])  # First part for training
            test_ids = set(subjects[train_count:])   # Remaining part for testing

            # Ensure there is no intersection between train and test sets
            assert train_ids.isdisjoint(test_ids), "Data leakage detected: Subject IDs are overlapping."

            x_train, y_train, x_test, y_test = [], [], [], []

            for subj_id in subjects:
                data = subj_to_data[subj_id]
                if subj_id in test_ids:
                    x_test.extend(data['scans'])
                    y_test.extend(data['ages'])
                else:
                    x_train.extend(data['scans'])
                    y_train.extend(data['ages'])

            # Print the shapes and types of data for debugging
            print(f"Train Scans: {len(x_train)}, Train Labels: {len(y_train)}")
            print(f"Test Scans: {len(x_test)}, Test Labels: {len(y_test)}")

            # Convert labels to consistent format
            y_train = self.format_labels(y_train)  
            y_test = self.format_labels(y_test)    

            # Print formatted labels for debugging
            print(f"Formatted Train Labels Shape: {y_train.shape}, Test Labels Shape: {y_test.shape}")  # Added for debugging

            yield np.array(x_train), np.array(x_test), y_train, y_test
    
    def __repr__(self):
        """
        Represents the Dataset instance as a string, providing basic information about the dataset.
        Returns:
            str: A string representation of the Dataset instance, including its name and the number of scans.
        """
        return f"<Dataset: {self.name} with {len(self.scanIDs)} scans>"

### Artificial Data ###

def num2vect(x: Union[float, np.ndarray], 
             bin_range: Tuple[float, float], 
             bin_step: float, 
             sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a number or an array of numbers to a vector representation based on specified binning 
    and Gaussian smoothing.

    Parameters:
    - x (float or np.ndarray): The input value or array of values to be converted.
    - bin_range (Tuple[float, float]): The range (start, end) of the bins.
    - bin_step (float): The step size between bins. Should evenly divide the range.
    - sigma (float): Standard deviation for Gaussian smoothing.
                     Use sigma=0 for hard labels, sigma>0 for soft labels.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - Vector representation of `x` as a NumPy array.
        - Array of bin centers.

    Raises:
    - ValueError: If bin_range is not divisible by bin_step or if bin_step is non-positive.
    """
    bin_start, bin_end = bin_range
    bin_length = bin_end - bin_start

    if bin_step <= 0 or bin_length % bin_step != 0:
        raise ValueError("bin_step must be positive and bin_range must be divisible by bin_step")

    bin_centers = np.linspace(bin_start + bin_step / 2, bin_end - bin_step / 2, int(bin_length / bin_step))

    if sigma == 0:
        indices = np.clip(np.floor((np.array(x) - bin_start) / bin_step), 0, len(bin_centers) - 1).astype(int)
        return indices, bin_centers

    v = np.zeros((np.size(x), len(bin_centers))) if np.size(x) > 1 else np.zeros(len(bin_centers))
    for i, center in enumerate(bin_centers):
        x1 = center - bin_step / 2
        x2 = center + bin_step / 2
        cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
        v[..., i] = np.diff(cdfs)

    return v.squeeze(), bin_centers

def crop_center(data: np.ndarray, 
                out_sp: Tuple[int, int, int]) -> np.ndarray:
    """
    Crop the center part of the volume data to a specified size.

    Parameters:
    - data (np.ndarray): The input volume data, which can be 3D or 4D.
    - out_sp (Tuple[int, int, int]): The desired output shape (x, y, z).

    Returns:
    - np.ndarray: The cropped volume data.

    Raises:
    - ValueError: If the dimension of `data` is not 3 or 4, or if the output shape is larger than the input shape.

    Example:
    >>> data = np.random.rand(182, 218, 182)
    >>> out_sp = (160, 192, 160)
    >>> data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = data.ndim

    if nd not in [3, 4]:
        raise ValueError(f"Wrong dimension! Expected 3 or 4 but got {nd}.")

    if any(i < o for i, o in zip(in_sp[-3: ], out_sp)):
        raise ValueError("Output shape cannot be larger than the input shape.")

    crop_slices = tuple(slice(int((i - o) / 2), int((i + o) / 2)) for i, o in zip(in_sp[-3: ], out_sp))
    
    return data[(slice(None), ) * (nd - 3) + crop_slices]

# def prepare_tensor(data: np.ndarray, 
#                    shape_prefix: Tuple[int, int] = (1, 1), 
#                    use_gpu: bool = True) -> tf.Tensor:
#     """
#     Reshape a NumPy array and convert it to a TensorFlow tensor. Optionally move the tensor to GPU.

#     Parameters:
#     - data (np.ndarray): The input NumPy array.
#     - shape_prefix (Tuple[int, int]): A tuple to prefix the shape of the array.
#     - use_gpu (bool): If True and a GPU is available, move the tensor to GPU.

#     Returns:
#     - tf.Tensor: The resulting TensorFlow tensor.
#     """
#     reshaped_data = data.reshape(shape_prefix + data.shape)
#     tensor = tf.convert_to_tensor(reshaped_data, dtype = tf.float32)

#     if use_gpu and tf.config.list_physical_devices('GPU'):
#         tensor = tensor.gpu()
#     elif use_gpu:
#         print("GPU not available, using CPU.")

#     return tensor

def prepare_tensor(data_list: List[np.ndarray], 
                   num_samples: int, 
                   use_gpu: bool = True) -> tf.Tensor:
    """
    Reshape a list of NumPy arrays and convert it to a TensorFlow tensor. Optionally move the tensor to GPU.

    Parameters:
    - data_list (List[np.ndarray]): The list of input NumPy arrays.
    - C (int): The size of the second dimension in the output tensor.
    - use_gpu (bool): If True and a GPU is available, move the tensor to GPU.

    Returns:
    - tf.Tensor: The resulting TensorFlow tensor of shape (num_samples, C, h, w, z).
    """
    # Validate that all arrays have the same shape
    shapes = [arr.shape for arr in data_list]
    
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All arrays in the list must have the same shape.")

    # Validate that C matches the desired shape
    if num_samples != len(shapes):
        raise ValueError("The value of num_samples must match the first dimension of the arrays in the list.")

    # Stack the arrays along a new dimension
    stacked_data = np.stack(data_list, axis=0)
    stacked_data = np.expand_dims(stacked_data, axis = 1)

    # Convert to TensorFlow tensor
    tensor = tf.convert_to_tensor(stacked_data, dtype=tf.float32)

    # Move to GPU if desired and available
    if use_gpu and tf.config.list_physical_devices('GPU'):
        tensor = tensor.gpu()
    elif use_gpu:
        print("GPU not available, using CPU.")

    return tensor

def generate_artificial_data(input_shape = (C.INPUT_SHAPE), num_samples = C.NUM_TRAIN_SAMPLES):
    
    bin_range = (42, 82)
    bin_step = 1
    sigma = 2
    
    X_list = []
    y_list = []

    for i in range(num_samples):

        X_init = np.random.rand(182, 218, 182)
        X_list.append(X_init)
        y_init = np.array([np.round(np.random.uniform(bin_range[0], bin_range[1]), 1)])
        # y_init = np.array([np.round(np.random.uniform(80, 80), 1)])
        y, bc = num2vect(y_init, bin_range, bin_step, sigma)
        print(y.shape)
        y_list.append(y)
    
    # Prepare the tensor for X
    X = prepare_tensor(data_list = X_list, num_samples = num_samples, use_gpu = True)
    X = tf.transpose(X, perm = [0, 2, 3, 4, 1])

    y = tf.convert_to_tensor(y_list, dtype = tf.float32)
    # y = tf.reshape(y, (num_samples, bin_range[1] - bin_range[0]))

    print(f"Type of X: {type(X)}, Shape of X: {X.shape}")
    print(f"Type of y: {type(y)}, Shape of y: {y.shape}")
    
    return X, y

def generate_artificial_data_constant_y(input_shape=(C.INPUT_SHAPE), num_samples=C.NUM_TRAIN_SAMPLES):
        
    X_list = []
    y_list = []

    for i in range(num_samples):

        X_init = np.random.rand(182, 218, 182)
        X_list.append(X_init)
        
        y_init = np.ones((40,))  # Adjust the shape of y_init if needed to match the original y's dimensions
        y_list.append(y_init)
    
    # Prepare the tensor for X
    X = prepare_tensor(data_list=X_list, num_samples=num_samples, use_gpu=True)
    X = tf.transpose(X, perm=[0, 2, 3, 4, 1])  

    # Convert y_list to a TensorFlow tensor
    y = tf.convert_to_tensor(y_list, dtype=tf.float32)
    # Reshape y if necessary to match the desired dimensions

    print(f"Type of X: {type(X)}, Shape of X: {X.shape}")
    print(f"Type of y: {type(y)}, Shape of y: {y.shape}")
    
    return X, y