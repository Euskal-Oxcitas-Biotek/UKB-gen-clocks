# Parameters
HEIGHT = 100
WIDTH = 100
NO_SLICES = 80
CHANNELS  = 1
INPUT_SHAPE = (NO_SLICES, HEIGHT, WIDTH, CHANNELS)

CSIZE = 128

MIN_AGE = 18
MAX_AGE = 98

# Hyperparameters
INITIAL_FILTERS = 16
L2_STRENGTH = 0.001
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 4
DEPTH = 4

NO_FOLDS = None
TEST_SIZE = 0.4

NUM_TRAIN_SAMPLES = 4  # Number of training samples
NUM_TEST_SAMPLES = 2    # Number of testing samples
SPLIT_PERC = 0.9        # Percentage for training set

_i = '08'

LOW_BIN = 55.0
HGH_BIN = 85.0
STEP_BIN = 1.0
SIGMA_BIN = 1.0

BINS = 5
ALPHA = 0.5
BETA = 0.5

HEALTHY_ORX = True
HEALTHY_COLE = False
PTGENDER = False
GENDER = 1

## LR Scheduler
EPOCHS_RED = 0.3
EPOCHS_MOD = 30
##

## Load weights
LOAD_WEIGHTS = False

## ResNet archs
RESNET = True
SOFT_LABELS = True