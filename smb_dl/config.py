import torch

SEED = 42  # check available device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '../data'
OGGM_WD = DATA_DIR + '/oggm_wd'
OGGM_OUT_DIR = DATA_DIR + '/oggm_out'
RESULTS_DIR = DATA_DIR + '/results/uq'

N_GLACIERS = 500  # how many glaciers to use (the largest have priority)
FP_SMB_DATA = OGGM_OUT_DIR + f'/oggm_smb_{N_GLACIERS}_glaciers.csv'
BATCH_SIZE = 512
MAX_N_EPOCHS = 6
Z_NOISE_LIST = [0.0, 0.1, 0.2, 0.3]
MODEL_TYPE_LIST = ['standard', 'gaussian']
DROPOUT_P_LIST = [0.0, 0.2]
NUM_SAMPLES_MC_DROPOUT = 100
NUM_SEEDS = 10
USE_DIFFERENT_SPLITS_PER_SEED = True
SHOW_PLOTS = True
SAVE_PLOTS = True
