import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from tqdm import tqdm
import joblib

from data_utils import add_gaussian_noise, train_valid_test_split
import config as local_cfg

if __name__ == '__main__':
    training_dir = Path(local_cfg.RESULTS_TRAINING_DIR) / 'baseline' / 'rf'
    inference_dir = Path(local_cfg.RESULTS_INFERENCE_DIR) / 'baseline' / 'rf'

    use_hpo = True

    params_grid = {
        'n_estimators': [100, 500],
        'max_features': ['sqrt', 1.0],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [10, 5, 2]
    }

    for z_noise in local_cfg.Z_NOISE_LIST:
        pbar = tqdm(range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS), desc=f'z_noise = {z_noise}')
        for seed_model in pbar:
            seed_split = seed_model

            # prepare the training data
            df = pd.read_csv(local_cfg.FP_SMB_DATA_PROCESSED)
            df = df.set_index(['gid', 'year'])

            # add noise
            df_noisy = add_gaussian_noise(df, min_z_noise=z_noise, max_z_noise=z_noise, seed=seed_model)

            print(f'\n\nSettings: seed_model = {seed_model}; seed_split = {seed_split}; z_noise = {z_noise}\n')
            print(df_noisy)
            df_train, df_valid, df_test, idx_train, idx_valid, idx_test = train_valid_test_split(
                df_annual=df_noisy,
                seed=seed_split
            )

            x_train = df_train.iloc[:, :-3].values
            y_train = df_train.annual_mb.values

            # check if the results already exist and skip if needed
            label = f'z_{z_noise:.2f}_seed_model_{seed_model}_seed_split_{seed_split}'
            fp_model = Path(training_dir) / 'models' / f'rf_model_{label}.pkl'
            if not fp_model.exists() or local_cfg.RETRAIN_MODEL_FORCE:
                if not use_hpo:
                    # build the model with the default parameters
                    print(f'Training RF with the default params')
                    rf_model = RandomForestRegressor(random_state=seed_model, n_jobs=-1)
                    rf_model.fit(x_train, y_train)
                else:
                    print('Running HPO')
                    # extract the validation data and run HPO using it
                    x_valid = df_valid.iloc[:, :-3].values
                    y_valid = df_valid.annual_mb.values

                    # use CV with a single split
                    split_index = [-1] * len(y_train) + [0] * len(y_valid)
                    x = np.concatenate((x_train, x_valid), axis=0)
                    y = np.concatenate((y_train, y_valid), axis=0)
                    pds = PredefinedSplit(test_fold=split_index)
                    hpo = GridSearchCV(
                        estimator=RandomForestRegressor(random_state=seed_model, n_jobs=8),
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
                    fp = Path(training_dir) / 'hpo_results' / f'hpo_results_{label}.csv'
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    cv_results_df.to_csv(fp, index=False)

                    # fit the model with the best parameters
                    print(f'Training RF with best params: {hpo.best_params_}')
                    rf_model = RandomForestRegressor(**hpo.best_params_, n_jobs=-1, random_state=seed_model)
                    rf_model.fit(x_train, y_train)

                # export the model
                fp_model.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(rf_model, filename=fp_model)
            else:
                print(f'{fp_model} already exists. Skipping the training.')
                # load the model
                rf_model = joblib.load(filename=fp_model)

            # compute MAE scores
            res_df_list = []

            for fold in ['train', 'valid', 'test']:
                for with_noise in [False, True]:
                    # get the data for the current fold
                    df_noisy_crt_fold = {'train': df_train, 'valid': df_valid, 'test': df_test}[fold]
                    x = df_noisy_crt_fold.iloc[:, :-3].values
                    y_true_col = 'annual_mb' if with_noise else 'annual_mb_orig'
                    y_true = df_noisy_crt_fold[y_true_col].values
                    noise = df_noisy_crt_fold['label_noise'].values

                    # get predictions
                    y_pred = rf_model.predict(x)
                    y_std = np.std(np.vstack([t.predict(x) for t in rf_model.estimators_]), axis=0)
                    mae_scores = np.abs(y_pred - y_true)
                    res_df = pd.DataFrame(
                        {
                            'fold': fold,
                            'idx': {'train': idx_train, 'valid': idx_valid, 'test': idx_test}[fold],
                            'with_noise': with_noise,
                            'noise': noise,
                            'y_true': y_true,
                            'y_pred': y_pred,
                            'y_std': y_std,
                            'mae': mae_scores,
                        }
                    )
                    res_df_list.append(res_df)

            res_df = pd.concat(res_df_list)
            print(f'MAE stats (z_noise = {z_noise}; seed_model = {seed_model}):')
            print(res_df.groupby(['fold', 'with_noise']).mae.describe())

            fp = Path(inference_dir) / 'stats' / f'stats_{label}.csv'
            fp.parent.mkdir(parents=True, exist_ok=True)
            res_df.to_csv(fp, index=False)
