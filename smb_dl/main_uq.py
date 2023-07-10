import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange, tqdm
from matplotlib import pyplot as plt
from pathlib import Path

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
):
    """Train model for Map estimate.

    Args:
      model: model to train
      criterion: loss function
      train_loader: dataloader with training data
      lr: learning rate
      n_epochs: number of epochs to train for

    Returns:
      trained model
    """

    scores_per_epoch = []
    model = model.to(local_cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(n_epochs)
    for i_epoch in pbar:
        scores = {f'train_{k}': [] for k in ['loss', 'MAE', 'RMSE']}
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X)
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
                if isinstance(criterion, NLL_LOSS):
                    loss = criterion(y_pred, y, crt_epoch=i_epoch)
                else:
                    loss = criterion(y_pred, y)
                scores['valid_loss'].append(loss.detach().cpu().item())
                mae = (y_pred[:, 0] - y[:, 0]).abs().mean().detach().cpu().item()
                scores['valid_MAE'].append(mae)
                mse = ((y_pred[:, 0] - y[:, 0]).detach().cpu() ** 2).mean().sqrt().cpu().item()
                scores['valid_RMSE'].append(mse)

        scores_per_epoch.append({k: np.mean(v) for (k, v) in scores.items()})
        pbar.set_postfix(scores='; '.join([f"{k}={v:.3f}" for (k, v) in scores_per_epoch[-1].items()]))

        # check the loss of the last n epochs compared to the previous n epochs, where n = patience
        if i_epoch > 2 * patience:
            avg_score_last_n = np.mean([x['valid_MAE'] for x in scores_per_epoch[-patience:]])
            avg_score_prev_last_n = np.mean([x['valid_MAE'] for x in scores_per_epoch[-(2 * patience): -patience]])
            if avg_score_last_n >= avg_score_prev_last_n - 1e-4:
                print(f'Early stop at epoch {i_epoch + 1}/{n_epochs}; '
                      f'avg_score_last_n = {avg_score_last_n:.3f}; '
                      f'avg_score_prev_last_n = {avg_score_prev_last_n:.3f}')
                break

    return scores_per_epoch


if __name__ == '__main__':
    for seed in range(local_cfg.SEED, local_cfg.SEED + local_cfg.NUM_SEEDS):
        for z_noise in local_cfg.Z_NOISE_LIST:
            for model_type in local_cfg.MODEL_TYPE_LIST:
                for dropout_p in local_cfg.DROPOUT_P_LIST:
                    df_annual = prepare_smb_df(fp=local_cfg.FP_SMB_DATA)
                    out_dir = Path(local_cfg.RESULTS_DIR) / model_type / f'dropout_p_{dropout_p}'

                    # add noise to the data
                    df_annual = add_gaussian_noise(df_annual, min_z_noise=z_noise, max_z_noise=z_noise, seed=seed)

                    print(f'\n\nSettings: seed = {seed}; z_noise = {z_noise}; model_type = {model_type}\n')
                    print(df_annual)
                    seed_splits = seed if local_cfg.USE_DIFFERENT_SPLITS_PER_SEED else local_cfg.SEED
                    print(f'seed_splits = {seed_splits}')
                    df_train, df_valid, df_test = train_valid_test_split(df_annual, seed=seed_splits)

                    x_train, y_train, x_valid, y_valid, x_test, y_test, = get_smb_data_tensors(
                        df_train, df_valid, df_test,
                        input_cols=list(df_annual.columns)[:-3],
                        output_col='annual_mb'
                    )

                    # scale the features
                    x_train, x_valid, x_test = standardize_data(df_train, x_train, x_valid, x_test)

                    # prepare the dataloaders
                    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=local_cfg.BATCH_SIZE)
                    valid_dl = DataLoader(TensorDataset(x_valid, y_valid), batch_size=len(x_valid))
                    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=len(x_test))

                    print(pd.Series(y_train.cpu().numpy().flatten()).describe())

                    # fix the seed for pytorch
                    torch.manual_seed(seed)

                    if model_type == 'standard':
                        model = MLP(n_inputs=x_train.shape[1], n_hidden=[50, 25], dropout_p=dropout_p)
                        loss = torch.nn.MSELoss()
                    else:
                        model = MLP(n_inputs=x_train.shape[1], n_hidden=[50, 25], n_outputs=2, predict_sigma=True,
                                    dropout_p=dropout_p)
                        loss = NLL_LOSS()

                    # train model
                    patience = 25 if dropout_p == 0.0 else 100
                    scores_per_epoch = train_model(
                        model, loss, train_dl, valid_dl, lr=2e-4, n_epochs=local_cfg.MAX_N_EPOCHS, patience=patience)
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
                            fn = f'learning_curve_z_{z_noise:.2f}_seed_{seed}.png'
                            fp = Path(out_dir) / fn
                            fp.parent.mkdir(parents=True, exist_ok=True)
                            plt.savefig(fp)
                        # plt.show()

                    # compute MAE scores
                    res_df_list = []
                    for fold in ['train', 'valid', 'test']:
                        for with_noise in [False, True]:
                            dl = {'train': train_dl, 'valid': valid_dl, 'test': test_dl}[fold]

                            n_samples = 1 if dropout_p == 0.0 else local_cfg.NUM_SAMPLES_MC_DROPOUT

                            # get predictions
                            X = dl.dataset.tensors[0]
                            outputs = torch.stack([model(X) for i in range(n_samples)], dim=2).detach()
                            y_pred = outputs[:, 0, :].mean(dim=1).cpu().numpy()
                            if model_type == 'gaussian':
                                sigma_pred = torch.exp(outputs[:, 1, :]).mean(dim=1).sqrt().cpu().numpy()
                            else:
                                sigma_pred = np.zeros_like(y_pred) + np.nan
                            y_true = dl.dataset.tensors[1].cpu().numpy().flatten()
                            df = {'train': df_train, 'valid': df_valid, 'test': df_test}[fold]
                            noise = df['label_noise'].values
                            if not with_noise:
                                y_true -= noise
                            mae_scores = np.abs(y_pred - y_true)
                            res_df = pd.DataFrame({
                                'fold': fold,
                                'with_noise': with_noise,
                                'noise': noise,
                                'y_true': y_true,
                                'y_pred': y_pred,
                                'sigma_pred': sigma_pred,
                                'mae': mae_scores
                            })
                            res_df_list.append(res_df)

                    res_df = pd.concat(res_df_list)
                    res_df_summary = res_df.groupby(['fold', 'with_noise']).mae.describe()
                    print('MAE stats:')
                    print(res_df_summary)
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    fp = Path(out_dir) / f'stats_z_{z_noise:.2f}_seed_{seed}.csv'
                    res_df.to_csv(fp, index=False)
                    fp = Path(out_dir) / f'stats_summary_z_{z_noise:.2f}_seed_{seed}.csv'
                    res_df_summary.to_csv(fp)

                    # save the model weights
                    fp = Path(out_dir) / f'model_weights_z_{z_noise:.2f}_seed_{seed}.pt'
                    torch.save(model.state_dict(), fp)
                    print(f'Model weights saved to {fp}')
