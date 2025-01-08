import os
import json
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import time

from src.data import Dataset
import src.constants as C
from src.models import Custom3DCNN, SFCN, resnet18
from src.utils import AWDLoss, KLDivLoss, LocalMAE, BinMAELogger, CustomMAE, CustomMSE
from src.utils import load_scan, load_model, MetricsLogger, save_subject_ids, SaveWeightsEveryNEpochs, log_json
from src.utils import discretize_ages, scheduler, PearsonCorrelation
from src.utils import augment, visualize_augmentations, SaveAugmentationCallback
from src.utils import format_time, calculate_representative_age, compute_alpha_beta
from src.utils import create_experiment_folder
from src.utils import CustomSiMAE, CustomSiMSE

def prepare_data(dataset):
    """
    Prepare the dataset by appending file extensions to scan IDs.
    """
    dataset.scanIDs = [scan + '.mgz' for scan in dataset.scanIDs]
    return dataset

def compile_model_real_labels(model, learning_rate):
    """
    Compile the model for real labels with the custom AWD loss function and Adam optimizer.
    """
    awdLoss = AWDLoss(num_bins=C.BINS, initial_alpha=C.ALPHA, initial_beta=C.BETA)
    
    def awd_loss(y_true, y_pred):
        return awdLoss(y_true, y_pred)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=awd_loss,
                  metrics=['mean_absolute_error', 'mean_squared_error', LocalMAE(num_bins=C.BINS)])
    return model

def compile_model_soft_labels(model, learning_rate = C.LEARNING_RATE):
    """
    Compile the model for soft labels with Kullback-Leibler Divergence loss and Adam optimizer.
    """
    # Learning rate and optimizer setup
    initial_learning_rate = learning_rate
    sgd = SGD(learning_rate=initial_learning_rate, momentum=0.9)
    
    model.compile(optimizer=sgd,
                  loss=KLDivLoss,
                  metrics=[CustomMAE(), CustomMSE(), LocalMAE(num_bins=C.BINS), PearsonCorrelation()])
    return model

def create_data_batches(X_train, y_train, X_test, y_test, batch_size, soft_labels=False):
    """
    Create TensorFlow data batches for training and testing.
    """
    if soft_labels:
        train_labels = np.array([item for item in y_train])
        test_labels = np.array([item for item in y_test])
    else:
        train_labels = np.array(y_train)
        test_labels = np.array(y_test)

    train_scans = tf.data.Dataset.from_tensor_slices((X_train, train_labels))
    test_scans = tf.data.Dataset.from_tensor_slices((X_test, test_labels))

    train_scans = train_scans.map(load_scan, num_parallel_calls=tf.data.AUTOTUNE).cache()
    test_scans = test_scans.map(load_scan, num_parallel_calls=tf.data.AUTOTUNE)
    
    BUFFER_SIZE = 500
    train_batches = (
        train_scans
        .cache()
        .shuffle(BUFFER_SIZE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .filter(lambda x, y: tf.shape(x)[0] > 1)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = test_scans.batch(batch_size).filter(lambda x, y: tf.shape(x)[0] > 1)

    return train_batches, test_batches

def main(folder_name):
    # Load training configuration
    config_path = Path('../configs/training_config.json').expanduser().resolve()
    print(f"Reading training configuration from: {config_path}")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Extract configuration parameters
    project_name = config['project_name']
    output_directory = Path(config['output_directory']).expanduser().resolve()
    use_soft_labels = config['use_soft_labels']
    use_soft_labels = C.SOFT_LABELS

    if use_soft_labels:
        output_file = output_directory / 'soft_folds_info.json'
        dataset_config_path = output_directory / 'configs' / 'dataset_config_soft.json'
        targets_json_file_path = output_directory / 'targets' / 'soft_targets.json'
    else:
        output_file = output_directory / 'real_folds_info.json'
        dataset_config_path = output_directory / 'configs' / 'dataset_config_real.json'
        targets_json_file_path = output_directory / 'targets' / 'real_targets.json'

    # Placeholder for running the actual experiment
    print(f"Running experiment with config: epochs={C.EPOCHS}, learning_rate={C.LEARNING_RATE}")

    # Load the dataset configuration from the file
    dataset = Dataset.from_file(dataset_config_path)
    dataset = prepare_data(dataset)

    start_time = time.time()
    for fold_num, (X_train, X_test, y_train, y_test) in enumerate(dataset.train_test_subjID_iterator(), 1):
        print(f"Training fold {fold_num}...")
        print(f"Number of cases in X_train: {len(X_train)}")
        print(f"Number of cases in X_test: {len(X_test)}")

        y_train_bins, train_bins = discretize_ages(y_train)
        y_test_bins, test_bins = discretize_ages(y_test)

        model_pick = C.RESNET
        if use_soft_labels:
            print("Soft labeling case initialized...")
            if model_pick:
                print("Using ResNet model...")
                model = resnet18(input_shape=(C.CSIZE, C.CSIZE, C.CSIZE, 1),
                         num_classes=int((C.HGH_BIN - C.LOW_BIN) / C.STEP_BIN), 
                         channel_size=[64, 64, 128, 256, 512], 
                         dropout=True)
            else:
                print("Using SFCN model...")
                model = SFCN(input_shape=(C.CSIZE, C.CSIZE, C.CSIZE, 1), 
                            channels=[32, 64, 128, 256, 256, 64], 
                            output_dim=int((C.HGH_BIN - C.LOW_BIN) / C.STEP_BIN), 
                            use_dropout=True, 
                            csize=C.CSIZE).build_model()
            model = compile_model_soft_labels(model, C.LEARNING_RATE)
            train_batches, test_batches = create_data_batches(X_train, y_train, X_test, y_test, C.BATCH_SIZE, soft_labels=True)
        else:
            print("Regression case initialized...")
            model = Custom3DCNN(input_shape=(C.CSIZE, C.CSIZE, C.CSIZE, 1),
                                depth=C.DEPTH, 
                                initial_filters=C.INITIAL_FILTERS, 
                                l2_strength=C.L2_STRENGTH).build_model()
            model = compile_model_real_labels(model, C.LEARNING_RATE)
            train_batches, test_batches = create_data_batches(X_train, y_train, X_test, y_test, C.BATCH_SIZE)

        STEPS_PER_EPOCH = len(X_train) // C.BATCH_SIZE
        checkpoint_dir_fold = os.path.join(folder_name, f"model_output/fold_{fold_num}")
        os.makedirs(checkpoint_dir_fold, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir_fold, f"best_weights_fold_{fold_num}.h5")

        if C.LOAD_WEIGHTS and os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}...")
            model.load_weights(checkpoint_path)

        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='min',
                                     save_best_only=True,
                                     verbose=1)
        
        lr_scheduler = LearningRateScheduler(scheduler)

        log_file_path = os.path.join(folder_name, 'logs', f'metrics_log_fold_{fold_num}.txt')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        metrics_logger = MetricsLogger(log_file_path=log_file_path)

        bin_mae_logger = BinMAELogger(num_bins=C.BINS, log_dir=checkpoint_dir_fold)

        save_subject_ids(checkpoint_dir_fold, fold_num, X_train, X_test)

        save_weights_callback = SaveWeightsEveryNEpochs(save_freq=5, save_dir=checkpoint_dir_fold)

        augmentation_preview_dir = Path(folder_name) / 'augmentation_preview'
        augmentation_preview_dir.mkdir(parents=True, exist_ok=True)
        save_augmentation_callback = SaveAugmentationCallback(model, train_batches, augmentation_preview_dir)

        callbacks = [checkpoint, metrics_logger, save_weights_callback, lr_scheduler]

        history = model.fit(train_batches,
                            epochs=C.EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=test_batches,
                            verbose=1,
                            callbacks=callbacks)

        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path} to compute validation preds.")
            model.load_weights(checkpoint_path)
        
        # Validation preds
        y_true = []
        for _, labels in test_batches:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        y_pred = model.predict(test_batches)
        y_pred = tf.nn.softmax(y_pred)
        y_pred_np = y_pred.numpy()

        y_true_list = y_true.tolist()
        y_pred_list = y_pred_np.tolist()

        deb_data = {
            'y_true': y_true_list,
            'y_pred': y_pred_list
        }

        with open(os.path.join(folder_name, 'validation_predictions.json'), 'w') as json_file:
            json.dump(deb_data, json_file)

        print("Saved y_true and y_pred to validation_predictions.json")

        log_json(fold_num=fold_num,
                 checkpoint_path=checkpoint_path,
                 X_train=X_train,
                 X_test=X_test,
                 history=history,
                 output_file=output_file)

        end_time = time.time()
        elapsed_time = end_time - start_time
        if C.NO_FOLDS == None:
            print(f"Training completed and took {format_time(elapsed_time)}")
        else:
            print(f"Fold {fold_num} completed.")
        
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    folder_name = sys.argv[1]
    main(folder_name)
