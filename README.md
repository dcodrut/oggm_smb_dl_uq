## Uncertainty-Aware Learning with Label Noise for Glacier Mass Balance Modelling

### Set up the python environment
`conda env create -f environment.yml`  
`conda activate smbuqenv`  
`cd smd_dl`

### Reproduce the results
The running settings are stored in `config.py`. You can increase `NUM_PROCS` (default = 1) to train the models in parallel.
By default, for each model, the experiments are repeated `NUM_SEEDS` times (default = 10) for each of the 6 scenarios (clean labels + 5 noise levels, see `Z_NOISE_LIST`).

1. `python main_mlp.py`: trains MLP models (all versions): 
   - with the current settings this script will train 1440 models:
     - MLP, MLP+NLL, MLP+MCD, MLP+NLL+MCD: each has 6 * 10 = 60 models, so 240 models in total;
     - Ensemble(MLP), Ensemble(MLP+NLL): each has 6 * 10 * 10 (see `ENSEMBLE_SIZE`), so 1200 models in total;
2. `python main_agg_ensemble.py`: once all the MLPs are trained, the predictions have to be aggregated for the ensemble models 
3. `python main_skl.py`: fits LR & RF baselines, including HPO for RF

By default, the results will be stored in `/output/results/final` as follows:
- `/training` (see `RESULTS_TRAINING_DIR`):
  - `/baseline`: pickled LR and RF models & csv files with the HPO results for RF 
  - `/mlp`: models' weights & learning curve plots for the MLP-based models
- `/inference_iid` (see `RESULTS_INFERENCE_DIR`): separated csv files containing the predictions for all models & all noise scenarios


