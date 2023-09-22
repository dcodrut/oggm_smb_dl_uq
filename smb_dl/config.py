Z_NOISE_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
import torch

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../data/'

RESULTS_TRAINING_DIR = DATA_DIR + '/output/results/final/training'
RESULTS_INFERENCE_DIR = DATA_DIR + '/output/results/final/inference_iid'

N_GLACIERS = 1000  # how many glaciers to use (the largest have priority)
FP_SMB_DATA_PROCESSED = DATA_DIR + f'/oggm_smb_{N_GLACIERS}_glaciers_processed.csv'
FP_SMB_DATA_PROCESSED_FOR_NORM = FP_SMB_DATA_PROCESSED
BATCH_SIZE = 128
MAX_N_EPOCHS = 6000
LEARNING_RATE = 0.0004
PATIENCE = 200
MODEL_TYPE_LIST = ['standard', 'gaussian']
DROPOUT_P = 0.2
NUM_SAMPLES_MC_DROPOUT_TRAINING = 20
NUM_SAMPLES_MC_DROPOUT_INFERENCE = 100
ENSEMBLE_SIZE = 10
NUM_SEEDS = 10
USE_DIFFERENT_SPLITS_PER_SEED = True
SHOW_PLOTS = True
SAVE_PLOTS = True
NUM_PROCS = 1
RETRAIN_MODEL_FORCE = False
INPUT_COLS = (
        [f'temp_{m:02d}' for m in range(1, 13)] +
        [f'prcp_{m:02d}' for m in range(1, 13)] +
        ['area', 'z_min', 'z_max', 'z_med', 'slope', 'aspect_sin', 'aspect_cos']  # drop the coords
)
