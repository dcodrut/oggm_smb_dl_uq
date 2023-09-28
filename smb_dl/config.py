import torch

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../data/'

RESULTS_TRAINING_DIR = DATA_DIR + '/output/results/final/training'
RESULTS_INFERENCE_DIR = DATA_DIR + '/output/results/final/inference_iid'

Z_NOISE_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # amplitudes of label noise
FP_SMB_DATA_PROCESSED = DATA_DIR + f'/oggm_smb_1000_glaciers_processed.csv'
FP_SMB_DATA_PROCESSED_FOR_NORM = FP_SMB_DATA_PROCESSED  # which data to use to standardize the data
BATCH_SIZE = 128
MAX_N_EPOCHS = 6000
LEARNING_RATE = 0.0004
PATIENCE = 200
MODEL_TYPE_LIST = ['standard', 'gaussian']  # MLP models types
DROPOUT_P = 0.2
NUM_SAMPLES_MC_DROPOUT_TRAINING = 20
NUM_SAMPLES_MC_DROPOUT_INFERENCE = 100
ENSEMBLE_SIZE = 10
NUM_SEEDS = 10  # how many times to repeat the experiments with different model initializations (and data splits)
USE_DIFFERENT_SPLITS_PER_SEED = True  # if true, for each repeated experiment, the data splits are different
SAVE_PLOTS = True  # whether to save the plots of the learning curves
SHOW_PLOTS = False  # whether to show the plots of the learning curves while training
NUM_PROCS = 1  # how many MLP models to train in parallel
NUM_PROCS_SKL = 48  # how many cores to use for fitting the RF models/perform HPO for RF
RETRAIN_MODEL_FORCE = False
INPUT_COLS = (
        [f'temp_{m:02d}' for m in range(1, 13)] +
        [f'prcp_{m:02d}' for m in range(1, 13)] +
        ['area', 'z_min', 'z_max', 'z_med', 'slope', 'aspect_sin', 'aspect_cos']  # drop the coords
)  # list of input features
