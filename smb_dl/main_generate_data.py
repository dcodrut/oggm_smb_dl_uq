from oggm import cfg, utils, tasks, workflow, graphics
from oggm.core import massbalance, flowline

cfg.initialize(logging_level='WARNING')

cfg.PARAMS['use_multiprocessing'] = True

import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

import config as local_cfg

wd = local_cfg.OGGM_WD
assert Path(wd).exists()

cfg.PATHS['working_dir'] = wd


def prepare_smb_df(fp):
    df = pd.read_csv(fp)

    # make it like a training data matrix
    df_annual = df.pivot(index=['gid', 'year'], columns=['month'], values=['temp', 'prcp'])

    # rename the columns
    df_annual.columns = [f'{col[0]}_{col[1]:02d}' for col in df_annual.columns]

    # add the remaining features
    right_df = df.groupby(['gid', 'year']).first().reset_index()[
        ['year', 'gid', 'lon', 'lat', 'area', 'z_min', 'z_max', 'z_med', 'slope', 'aspect', 'mb']]
    df_annual = df_annual.reset_index().merge(right_df, how='left').rename(columns={'mb': 'annual_mb'})

    # use sin(aspect) and cos(aspect) as features
    i_col = list(df_annual.columns).index('aspect')
    df_annual.insert(loc=i_col+1, column='aspect_sin', value=np.sin(df_annual.aspect.values))
    df_annual.insert(loc=i_col+2, column='aspect_cos', value=np.cos(df_annual.aspect.values))
    del df_annual['aspect']

    # mm to m
    df_annual.loc[:, 'annual_mb'] /= 1000

    return df_annual


if __name__ == '__main__':
    # download RGI outlines
    utils.get_rgi_dir(version='62')

    fp = utils.get_rgi_region_file('11', version='62')
    gdf_all = gpd.read_file(fp)
    gdf_all = gdf_all.sort_values('Area', ascending=False)

    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 80

    overwrite_gdirs = False

    num_glaciers = 500  # how many glaciers to simulate (the largest in the region will be used)
    crt_k = 1
    i = 0
    pbar = tqdm(total=num_glaciers)
    while crt_k <= num_glaciers:
        if i == len(gdf_all):
            break
        gid = gdf_all.RGIId.iloc[i]
        i += 1
        print(f'k = {crt_k}; gid = {gid}')
        sgdf = gdf_all[gdf_all.RGIId.isin([gid])]
        print(sgdf)

        # download level-3 data if needed
        if overwrite_gdirs:
            gdirs = workflow.init_glacier_directories(sgdf, from_prepro_level=3)
        else:
            gdirs = workflow.init_glacier_directories(sgdf)
        print(gdirs)
        gdir = gdirs[0]

        # get the climate data
        fp_clim = Path(gdir.get_filepath('climate_historical'))
        ds_ch = xr.open_dataset(fp_clim)

        # build the MB model (skipp the current glacier if it fails)
        try:
            mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
        except Exception as e:
            print(e)
            continue

        # compute the specific MB for both the original input data and the noisy version
        fls = gdir.read_pickle('model_flowlines')
        y0, y1 = 1980, 2020
        years = np.arange(y0, y1)
        mb_ts_orig = mbmod.get_specific_mb(fls=fls, year=years)

        ds_ch_sel = ds_ch.where(ds_ch.time >= np.datetime64(f'{y0 - 1}-10-01'), drop=True)
        dates = [pd.to_datetime(d, format='%Y-%m-d') for d in ds_ch_sel.time.values]
        years_fs = mbmod.flowline_mb_models[0].years
        df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in dates],
            'year': years_fs[years_fs >= years[0]],
            'month': [d.month for d in dates],
            'temp': ds_ch_sel['temp'].values,
            'prcp': ds_ch_sel['prcp'].values,
            'mb': np.atleast_2d(mb_ts_orig).T.repeat(12, 0).flatten()
        })

        # add some features from RGI
        df['lon'] = sgdf.CenLon.iloc[0]
        df['lat'] = sgdf.CenLat.iloc[0]
        df['area'] = sgdf.Area.iloc[0]
        df['z_min'] = sgdf.Zmin.iloc[0]
        df['z_max'] = sgdf.Zmax.iloc[0]
        df['z_med'] = sgdf.Zmed.iloc[0]
        df['slope'] = sgdf.Slope.iloc[0]
        df['aspect'] = sgdf.Aspect.iloc[0]

        fp = Path(f'{local_cfg.OGGM_OUT_DIR}/per_glacier/{gid}.csv')
        fp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fp, index=False)

        crt_k += 1
        pbar.update(1)

    fp_all = sorted(list(Path(f'{local_cfg.OGGM_OUT_DIR}/per_glacier').glob('*.csv')))
    all_df = []
    for fp in fp_all:
        _df = pd.read_csv(fp)
        _df['gid'] = fp.stem
        all_df.append(_df)
    df = pd.concat(all_df)
    df.to_csv(local_cfg.FP_SMB_DATA_RAW, index=False)
    print(f'Data (raw) for all glaciers exported to {local_cfg.FP_SMB_DATA_RAW}')

    df = prepare_smb_df(local_cfg.FP_SMB_DATA_RAW)
    df.to_csv(local_cfg.FP_SMB_DATA_PROCESSED, index=False)
    print(f'Data (processed) for all glaciers exported to {local_cfg.FP_SMB_DATA_PROCESSED}')
