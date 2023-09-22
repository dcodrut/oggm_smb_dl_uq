import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import multiprocessing

import config as local_cfg
from data_utils import *
from models import NLL_LOSS, MLP


def train_model(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        n_epochs: int = 100,
        patience=10,
        save_best_model=False,
        save_last_model=False,
        out_dir=None,
        label='',
        device=local_cfg.DEVICE,
):
    """
    :param model: model to be trained
    :param criterion: loss function
    :param train_loader: training data loader
    :param valid_loader: validation data loader
    :param lr: learning rate
    :param n_epochs: maximum number of epochs
    :param patience: how many epoch to wait until the training stops if no improvement occurs on the validation set
    :param save_best_model: whether to save the best model (based on the validation set)
    :param save_last_model: whether to save the last model
    :param out_dir: the directory path where to save the model's weights
    :param label: a label to be added in the filename when saving the model's weights
    :param device: device to train on (e.g. torch.device('cpu'), torch.device('gpu'), torch.device('gpu:1') etc.)
    :return: the scores per epoch (can be used for plotting the learning curves)
    """

    scores_per_epoch = []
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_epochs)
    for i_epoch in pbar:
        scores = {f'train_{k}': [] for k in ['loss', 'MAE', 'RMSE']}
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X)

            # perform multiple forward passes if needed
            num_samples = local_cfg.NUM_SAMPLES_MC_DROPOUT_TRAINING if model.dropout_p > 0 else 1
            for _ in range(num_samples - 1):
                y_pred += model(X)
            y_pred /= num_samples

            if isinstance(criterion, NLL_LOSS):
                loss = criterion(y_pred, y, crt_epoch=i_epoch)
            else:
                loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scores['train_loss'].append(loss.detach().cpu().item())
            mae = (y_pred[:, 0] - y[:, 0]).abs().mean().detach().cpu().item()
            scores['train_MAE'].append(mae)
            mse = ((y_pred[:, 0] - y[:, 0]).detach().cpu() ** 2).mean().sqrt().cpu().item()
            scores['train_RMSE'].append(mse)

        scores.update({f'valid_{k}': [] for k in ['loss', 'MAE', 'RMSE']})
        with torch.no_grad():
            for X, y in valid_loader:
                y_pred = model(X)

                # perform multiple forward passes if needed
                for _ in range(num_samples - 1):
                    y_pred += model(X)
                y_pred /= num_samples

                if isinstance(criterion, NLL_LOSS):
                    loss = criterion(y_pred, y, crt_epoch=i_epoch)
                else:
                    loss = criterion(y_pred, y)
                scores['valid_loss'].append(loss.detach().cpu().item())
                mae = (y_pred[:, 0] - y[:, 0]).abs().mean().detach().cpu().item()
                scores['valid_MAE'].append(mae)
                mse = ((y_pred[:, 0] - y[:, 0]).detach().cpu() ** 2).mean().sqrt().cpu().item()
                scores['valid_RMSE'].append(mse)

                # check if the error reduced and then save the model
                if save_best_model and i_epoch > 1:
                    best_mae_so_far = np.mean([x['valid_MAE'] for x in scores_per_epoch[:-1]])
                    if mae < best_mae_so_far:
                        fp_model = Path(out_dir) / 'models' / f'model_weights_{label}_best.pt'
                        fp_model.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), fp_model)

        scores_per_epoch.append({k: np.mean(v) for (k, v) in scores.items()})
        pbar.set_postfix(scores='; '.join([f"{k}={v:.3f}" for (k, v) in scores_per_epoch[-1].items()]))

        # check the loss of the last n epochs compared to the previous n epochs, where n = patience
        # if the model is trained with the NLL loss and fixed sigma, ignore the first epochs
        if isinstance(criterion, NLL_LOSS) and criterion.max_epochs_ct_sigma is not None:
            if i_epoch < criterion.max_epochs_ct_sigma:
                continue

        if i_epoch > 2 * patience:
            avg_score_last_n = np.mean([x['valid_MAE'] for x in scores_per_epoch[-patience:]])
            avg_score_prev_last_n = np.mean([x['valid_MAE'] for x in scores_per_epoch[-(2 * patience): -patience]])
            if avg_score_last_n >= avg_score_prev_last_n - 1e-4:
                print(f'Early stop at epoch {i_epoch + 1}/{n_epochs}; '
                      f'avg_score_last_n = {avg_score_last_n:.3f}; '
                      f'avg_score_prev_last_n = {avg_score_prev_last_n:.3f}')
                break

    if save_last_model:
        fp_model = Path(out_dir) / 'models' / f'model_weights_{label}_last.pt'
        fp_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fp_model)

    return scores_per_epoch


def run_experiment(seed_model, seed_split, model_type, z_noise, dropout_p, experiment_name='', device=local_cfg.DEVICE):
    df_annual = pd.read_csv(local_cfg.FP_SMB_DATA_PROCESSED)
    df_annual = df_annual.set_index(['gid', 'year'])
    training_dir = Path(local_cfg.RESULTS_TRAINING_DIR) / experiment_name / model_type
    inference_dir = Path(local_cfg.RESULTS_INFERENCE_DIR) / experiment_name / model_type

    # add noise to the data
    df_annual = add_gaussian_noise(df_annual, min_z_noise=z_noise, max_z_noise=z_noise, seed=seed_model)

    print(f'\n\nSettings: seed_model = {seed_model}; seed_split = {seed_split};'
          f' z_noise = {z_noise}; model_type = {model_type}\n')
    print(df_annual)
    df_train, df_valid, df_test, idx_train, idx_valid, idx_test = train_valid_test_split(df_annual, seed=seed_split)

    x_train, y_train, x_valid, y_valid, x_test, y_test, = get_smb_data_tensors(
        df_train, df_valid, df_test,
        input_cols=local_cfg.INPUT_COLS,
        output_col='annual_mb',
        device=device
    )

    # prepare the dataframe used for normalizing the features
    df_train_for_norm = pd.read_csv(local_cfg.FP_SMB_DATA_PROCESSED_FOR_NORM).iloc[idx_train][local_cfg.INPUT_COLS]

    # scale the features
    x_train, x_valid, x_test = standardize_data(df_train_for_norm, x_train, x_valid, x_test)

    # prepare the dataloaders
    train_dl = DataLoader(
        TensorDataset(x_train, y_train), batch_size=local_cfg.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dl = DataLoader(
        TensorDataset(x_valid, y_valid), batch_size=local_cfg.BATCH_SIZE, num_workers=0, shuffle=True)
    test_dl = DataLoader(
        TensorDataset(x_test, y_test), batch_size=local_cfg.BATCH_SIZE, num_workers=0, shuffle=True)

    print(pd.Series(y_train.cpu().numpy().flatten()).describe())

    # fix the seed for pytorch
    torch.manual_seed(seed_model)

    if model_type == 'standard':
        model = MLP(n_inputs=x_train.shape[1], n_hidden=[50, 25], dropout_p=dropout_p)
        loss = torch.nn.MSELoss()
    else:
        model = MLP(n_inputs=x_train.shape[1], n_hidden=[50, 25], n_outputs=2, predict_sigma=True, dropout_p=dropout_p)
        loss = NLL_LOSS()

    # check if the results already exist and skip if needed
    label = f'z_{z_noise:.2f}_seed_model_{seed_model}_seed_split_{seed_split}'
    fp_model = Path(training_dir) / 'models' / f'model_weights_{label}_last.pt'
    if not fp_model.exists() or local_cfg.RETRAIN_MODEL_FORCE:
        # train the model
        params = dict(
            model=model,
            criterion=loss,
            train_loader=train_dl,
            valid_loader=valid_dl,
            lr=local_cfg.LEARNING_RATE,
            n_epochs=local_cfg.MAX_N_EPOCHS,
            patience=local_cfg.PATIENCE,
            save_best_model=True,
            save_last_model=True,
            out_dir=training_dir,
            label=label,
            device=device
        )
        print(f'Training model with params: {params}')
        scores_per_epoch = train_model(**params)
        scores_per_epoch_df = pd.DataFrame.from_records(scores_per_epoch)
        if local_cfg.SHOW_PLOTS:
            n_cols = len(scores_per_epoch_df.columns)
            plt.figure(figsize=(20, 4), dpi=120)
            for i, k in enumerate(scores_per_epoch_df.columns):
                plt.subplot(1, n_cols // 2, i % 3 + 1)
                plt.plot(np.arange(len(scores_per_epoch_df)) + 1, scores_per_epoch_df[k], linewidth=2,
                         label=k)
                plt.xlabel('epoch')
                plt.ylabel(k)
                plt.legend()
                if i < n_cols // 2:
                    plt.grid()
            if local_cfg.SAVE_PLOTS:
                fn = f'learning_curve_{label}.png'
                fp = Path(training_dir) / 'learning_curves' / fn
                fp.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fp)
            # plt.show()
    else:
        print(f'{fp_model} already exists. Skipping the training.')

    # load the best model
    fp_model = Path(training_dir) / 'models' / f'model_weights_{label}_best.pt'
    model.load_state_dict(torch.load(fp_model, map_location=device))
    model.to(device)

    # compute MAE scores
    res_df_list = []
    for fold in ['train', 'valid', 'test']:
        for with_noise in [False, True]:
            dl = {'train': train_dl, 'valid': valid_dl, 'test': test_dl}[fold]

            n_samples = 1 if dropout_p == 0.0 else local_cfg.NUM_SAMPLES_MC_DROPOUT_INFERENCE

            # get predictions
            x = dl.dataset.tensors[0]
            with torch.no_grad():
                outputs = torch.stack([model(x) for _ in range(n_samples)], dim=2).detach()
            y_pred = outputs[:, 0, :].mean(dim=1).cpu().numpy()
            y_true = dl.dataset.tensors[1].cpu().numpy().flatten()
            df = {'train': df_train, 'valid': df_valid, 'test': df_test}[fold]
            idx = {'train': idx_train, 'valid': idx_valid, 'test': idx_test}[fold]
            noise = df['label_noise'].values
            if not with_noise:
                y_true -= noise
            mae_scores = np.abs(y_pred - y_true)
            res_df = pd.DataFrame({
                'fold': fold,
                'idx': idx,
                'with_noise': with_noise,
                'noise': noise,
                'y_true': y_true,
                'y_pred': y_pred,
                'mae': mae_scores
            })

            # add the predicted sigmas if available
            if model_type == 'gaussian':
                sigma_pred = torch.exp(outputs[:, 1, :]).mean(dim=1).sqrt().cpu().numpy()
                res_df['sigma_pred'] = sigma_pred

            # add the standard deviation over the predictions is dropout is used
            if dropout_p != 0:
                y_std = outputs[:, 0, :].std(dim=1).cpu().numpy()
                res_df['y_std'] = y_std

            res_df_list.append(res_df)

    res_df = pd.concat(res_df_list)
    print('MAE stats:')
    print(res_df.groupby(['fold', 'with_noise']).mae.describe())
    fp = Path(inference_dir) / 'stats' / f'stats_{label}.csv'
    fp.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(fp, index=False)


def run_experiment_star(settings):
    return run_experiment(**settings)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    all_settings = []

    # 1. baseline MLP
    settings_baseline_mlp = []
    for crt_seed_model in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
        crt_seed_split = crt_seed_model if local_cfg.USE_DIFFERENT_SPLITS_PER_SEED else local_cfg.SEED
        for crt_z_noise in local_cfg.Z_NOISE_LIST:
            crt_settings = {
                'experiment_name': 'mlp',
                'seed_model': crt_seed_model,
                'seed_split': crt_seed_split,
                'z_noise': crt_z_noise,
                'model_type': 'standard',
                'dropout_p': 0.0
            }
            settings_baseline_mlp.append(crt_settings)
    print(f'len(settings_baseline_mlp) = {len(settings_baseline_mlp)}')

    # 2. Gaussian MLP
    settings_gaussian_mlp = []
    for crt_seed_model in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
        crt_seed_split = crt_seed_model if local_cfg.USE_DIFFERENT_SPLITS_PER_SEED else local_cfg.SEED
        for crt_z_noise in local_cfg.Z_NOISE_LIST:
            crt_settings = {
                'experiment_name': 'mlp',
                'seed_model': crt_seed_model,
                'seed_split': crt_seed_split,
                'z_noise': crt_z_noise,
                'model_type': 'gaussian',
                'dropout_p': 0.0
            }
            settings_gaussian_mlp.append(crt_settings)
    print(f'len(settings_gaussian_mlp) = {len(settings_gaussian_mlp)}')

    # 3. MC Dropout - both with and without Gaussian output
    settings_mc_dropout = []
    for crt_seed_model in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
        crt_seed_split = crt_seed_model if local_cfg.USE_DIFFERENT_SPLITS_PER_SEED else local_cfg.SEED
        for crt_z_noise in local_cfg.Z_NOISE_LIST:
            for crt_model_type in local_cfg.MODEL_TYPE_LIST:
                crt_settings = {
                    'experiment_name': 'mc_dropout',
                    'seed_model': crt_seed_model,
                    'seed_split': crt_seed_split,
                    'z_noise': crt_z_noise,
                    'model_type': crt_model_type,
                    'dropout_p': local_cfg.DROPOUT_P
                }
                settings_mc_dropout.append(crt_settings)
    print(f'len(settings_mc_dropout) = {len(settings_mc_dropout)}')

    # 4. Ensemble - both with and without Gaussian output
    settings_ensemble = []
    for crt_seed_split in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
        for crt_seed_model in range(local_cfg.SEED, local_cfg.SEED + local_cfg.ENSEMBLE_SIZE):
            for crt_z_noise in local_cfg.Z_NOISE_LIST:
                for crt_model_type in local_cfg.MODEL_TYPE_LIST:
                    crt_settings = {
                        'experiment_name': 'ensemble_members',
                        'seed_model': crt_seed_model,
                        'seed_split': crt_seed_split,
                        'z_noise': crt_z_noise,
                        'model_type': crt_model_type,
                        'dropout_p': 0.0
                    }
                    settings_ensemble.append(crt_settings)
    print(f'len(settings_ensemble) = {len(settings_ensemble)}')

    all_settings = settings_baseline_mlp + settings_gaussian_mlp + settings_mc_dropout + settings_ensemble
    print(f'len(all_settings) = {len(all_settings)}')
    print(f'#procs = {local_cfg.NUM_PROCS}')

    num_procs = local_cfg.NUM_PROCS
    if num_procs == 1:
        for crt_settings in tqdm(all_settings):
            run_experiment_star(crt_settings)
    else:
        with multiprocessing.Pool(num_procs) as pool:
            for _ in tqdm(pool.imap(run_experiment_star, all_settings, chunksize=1),
                          total=len(all_settings),
                          position=1):
                pass
