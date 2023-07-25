import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from tqdm import tqdm

import config as local_cfg

if __name__ == '__main__':
    res_dir_root = Path(local_cfg.RESULTS_DIR)

    out_dir = res_dir_root / 'baseline' / 'rf'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # read the training data
    data_df = pd.read_csv(local_cfg.FP_SMB_DATA_PROCESSED)
    data_df = data_df.set_index(['gid', 'year'])

    use_hpo = False
    params_grid = {
        'n_estimators': [100, 500],
        'max_features': ['sqrt', 1.0],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [10, 5, 2]
    }

    for z_score in local_cfg.Z_NOISE_LIST:
        pbar = tqdm(range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS), desc=f'z_score = {z_score}')
        for seed_model in pbar:
            seed_split = seed_model
            out_dir_baseline = res_dir_root / 'baseline' / 'standard'
            fn = f'stats_z_{z_score:.2f}_seed_model_{seed_model}_seed_split_{seed_split}.csv'
            fp = out_dir_baseline / fn
            res_df_mlp = pd.read_csv(fp)

            # prepare the training data using the same splits as for the MLP
            res_df_mlp_train = res_df_mlp[(res_df_mlp.fold == 'train') & res_df_mlp.with_noise]
            df_train = data_df.iloc[res_df_mlp_train.idx]
            x_train = df_train.iloc[:, :-1].values
            y_train = res_df_mlp_train.y_true.values

            if not use_hpo:
                # build the model with the default parameters
                print(f'Training RF with the default params')
                rf_model = RandomForestRegressor(random_state=seed_model, n_jobs=-1)
                rf_model.fit(x_train, y_train)
            else:
                print('Running HPO')
                # extract the validation data and run HPO using it
                res_df_mlp_valid = res_df_mlp[(res_df_mlp.fold == 'valid') & res_df_mlp.with_noise]
                df_valid = data_df.iloc[res_df_mlp_valid.idx]
                x_valid = df_valid.iloc[:, :-1].values
                y_valid = res_df_mlp_valid.y_true.values

                # use CV with a single split
                split_index = [-1] * len(y_train) + [0] * len(y_valid)
                x = np.concatenate((x_train, x_valid), axis=0)
                y = np.concatenate((y_train, y_valid), axis=0)
                pds = PredefinedSplit(test_fold=split_index)
                hpo = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=seed_model, n_jobs=-1),
                    param_grid=params_grid,
                    scoring='neg_mean_absolute_error',
                    verbose=True,
                    n_jobs=-1,
                    cv=pds,
                    return_train_score=True,
                    refit=False,
                )
                hpo.fit(x, y)

                # export the results
                cv_results_df = pd.DataFrame(hpo.cv_results_).sort_values('rank_test_score')
                print(cv_results_df)
                fp = Path(out_dir) / f'hpo_results_z_{z_score:.2f}_seed_model_{seed_model}_seed_split_{seed_split}.csv'
                cv_results_df.to_csv(fp, index=False)

                # fit the model with the best parameters
                print(f'Training RF with best params: {hpo.best_params_}')
                rf_model = RandomForestRegressor(**hpo.best_params_, n_jobs=-1, random_state=seed_model)
                rf_model.fit(x_train, y_train)

            # compute MAE scores
            res_df_list = []
            for fold in ['train', 'valid', 'test']:
                for with_noise in [False, True]:
                    # get the data for the current fold
                    res_df_mlp_crt_fold = res_df_mlp[(res_df_mlp.fold == fold) & (res_df_mlp.with_noise == with_noise)]
                    idx = res_df_mlp_crt_fold.idx
                    x = data_df.iloc[idx, :-1].values
                    y_true = res_df_mlp_crt_fold.y_true.values

                    # get predictions
                    y_pred = rf_model.predict(x)
                    noise = res_df_mlp_crt_fold.noise.values
                    mae_scores = np.abs(y_pred - y_true)
                    res_df = pd.DataFrame(
                        {
                            'fold': fold,
                            'idx': idx,
                            'with_noise': with_noise,
                            'noise': noise,
                            'y_true': y_true,
                            'y_pred': y_pred,
                            'mae': mae_scores,
                        }
                    )
                    res_df_list.append(res_df)

            res_df = pd.concat(res_df_list)
            res_df_summary = res_df.groupby(['fold', 'with_noise']).mae.describe()
            print(f'MAE stats (z_score = {z_score}; seed_model = {seed_model}):')
            print(res_df_summary)

            fp = Path(out_dir) / f'stats_z_{z_score:.2f}_seed_model_{seed_model}_seed_split_{seed_split}.csv'
            res_df.to_csv(fp, index=False)
            fp = Path(out_dir) / f'stats_summary_z_{z_score:.2f}_seed_model_{seed_model}_seed_split_{seed_split}.csv'
            res_df_summary.to_csv(fp)
