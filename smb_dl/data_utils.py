import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import torch

import config as local_cfg


def add_gaussian_noise(df_annual, min_z_noise: float = 0.1, max_z_noise: float = 0.1, seed=local_cfg.SEED):
    # add gaussian noise to training
    z_noise = np.linspace(max_z_noise, min_z_noise, num=len(df_annual))
    _scale = z_noise * np.std(df_annual.annual_mb)
    rng = np.random.default_rng(seed)
    label_noise = rng.normal(loc=0, scale=_scale)
    df_annual['annual_mb_orig'] = df_annual.annual_mb
    df_annual['label_noise'] = label_noise
    df_annual.annual_mb += label_noise
    return df_annual


def train_valid_test_split(df_annual, test_fraction=0.2, valid_fraction=0.1, seed=local_cfg.SEED):
    all_glaciers_ids = df_annual.reset_index().gid.values
    idx_train_valid, idx_test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            df_annual, groups=all_glaciers_ids
        )
    )
    df_train_valid = df_annual.iloc[idx_train_valid]

    idx_train, idx_valid = next(
        GroupShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed).split(
            df_train_valid, groups=all_glaciers_ids[idx_train_valid]
        )
    )

    df_train = df_train_valid.iloc[idx_train]
    df_valid = df_train_valid.iloc[idx_valid]
    df_test = df_annual.iloc[idx_test]

    # sanity checks
    assert len(set(df_train.reset_index().gid.values) & set(df_valid.reset_index().gid.values)) == 0
    assert len(set(df_train.reset_index().gid.values) & set(df_test.reset_index().gid.values)) == 0

    return df_train, df_valid, df_test, idx_train_valid[idx_train], idx_train_valid[idx_valid], idx_test


def get_smb_data_tensors(df_train, df_valid, df_test, input_cols, output_col):
    x_train = df_train[input_cols].values
    y_train = df_train[output_col].values
    x_valid = df_valid[input_cols].values
    y_valid = df_valid[output_col].values
    x_test = df_test[input_cols].values
    y_test = df_test[output_col].values

    # switch to tensors
    x_train = torch.Tensor(x_train).to(local_cfg.DEVICE)
    y_train = torch.Tensor(y_train).to(local_cfg.DEVICE).unsqueeze(1)
    x_valid = torch.Tensor(x_valid).to(local_cfg.DEVICE)
    y_valid = torch.Tensor(y_valid).to(local_cfg.DEVICE).unsqueeze(1)
    x_test = torch.Tensor(x_test).to(local_cfg.DEVICE)
    y_test = torch.Tensor(y_test).to(local_cfg.DEVICE).unsqueeze(1)

    # data loaders
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def standardize_data(df_train, x_train, x_valid, x_test):
    # scale the data in groups, to keep the relative distance
    cols = list(df_train.columns)
    f_groups = [
        [c for c in cols if 'temp' in c],
        [c for c in cols if 'prcp' in c],
        ['lon'], ['lat'], ['area'], ['z_min', 'z_med', 'z_max'], ['slope']
    ]

    for f_group in f_groups:
        idx = [cols.index(c) for c in f_group]
        _mu, _std = x_train[:, idx].mean(), x_train[:, idx].std()

        x_train[:, idx] -= _mu
        x_train[:, idx] /= _std

        x_valid[:, idx] -= _mu
        x_valid[:, idx] /= _std

        x_test[:, idx] -= _mu
        x_test[:, idx] /= _std

    return x_train, x_valid, x_test
