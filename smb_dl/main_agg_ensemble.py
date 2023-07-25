import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config as local_cfg

if __name__ == '__main__':
    res_dir_root = Path(local_cfg.RESULTS_DIR)

    for model_name in local_cfg.MODEL_TYPE_LIST:
        res_dir = res_dir_root / 'ensemble_members' / model_name

        for z_score in local_cfg.Z_NOISE_LIST:
            pbar = tqdm(range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS),
                        desc=f'model_name = {model_name}; z_score = {z_score}')
            for seed_split in pbar:
                # read the outputs from all ensemble members
                res_df_list = []
                for seed_model in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
                    fp = res_dir / f'stats_z_{z_score:.2f}_seed_model_{seed_model}_seed_split_{seed_split}.csv'
                    res_df = pd.read_csv(fp)
                    res_df['seed_model'] = seed_model
                    res_df_list.append(res_df)

                # aggregate the predictions
                res_df = pd.concat(res_df_list)
                agg_f = {
                    'fold': 'first',
                    'noise': 'first',
                    'y_true': 'first',
                    'y_pred': [np.mean, np.std],
                }
                if 'sigma_pred' in res_df.columns:
                    agg_f['sigma_pred'] = lambda x: np.sqrt(np.mean(x ** 2))
                res_df = res_df.groupby(['idx', 'with_noise'], sort=False).agg(agg_f).reset_index()
                new_cols = ['idx', 'with_noise', 'fold', 'noise', 'y_true', 'y_pred', 'y_std']
                if 'sigma_pred' in res_df.columns:
                    new_cols += ['sigma_pred']
                res_df.columns = new_cols

                # compute MAE scores based on the average predictions
                res_df['mae'] = (res_df.y_pred - res_df.y_true).abs()

                res_df_summary = res_df.groupby(['fold', 'with_noise']).mae.describe()
                print(f'MAE - ensemble (model_name = {model_name}; z_score = {z_score}; seed_split = {seed_split}):')
                print(res_df_summary)

                out_dir = res_dir_root / 'ensemble' / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                fp = out_dir / f'stats_z_{z_score:.2f}_seed_model_all_seed_split_{seed_split}.csv'
                res_df.to_csv(fp, index=False)
                fp = out_dir / f'stats_summary_z_{z_score:.2f}_seed_model_all_seed_split_{seed_split}.csv'
                res_df_summary.to_csv(fp)
