import numpy as np
import xarray as xr
import os

year = 2012
month = 12

month_datasets = []

prefix = "/g/data/gb6/BRAN/BRAN2020/month"
#prefix = os.path.expanduser("~/geoscience/data")
month_clims = []

for month in range(1, 13):
    for year in range(1993, 2020):
        print('Processing', month, year)
        temp_fp = os.path.join(prefix, f"ocean_temp_mth_{year}_{month:02d}.nc")
        temp_da = xr.load_dataset(temp_fp).temp

        mld_fp = os.path.join(prefix, f"ocean_mld_mth_{year}_{month:02d}.nc")
        mld_da = xr.load_dataset(mld_fp).mld

        mld = mld_da.data.ravel()
        temp = temp_da.data.reshape((51, -1))
        depth = temp_da.st_ocean.data

        nanmask = ~np.isnan(mld)
        _temp = temp[:, nanmask]
        _mld = mld[nanmask]

        mask = depth[None, :] <= _mld[:, None]
        latlon_idxs, depth_idxs = np.where(np.diff(mask, axis=1))

        dsst = _temp[depth_idxs, latlon_idxs] - _temp[depth_idxs + 1, latlon_idxs]

        midlayer_mask = (depth[:, None] > _mld[None, :]) & (depth[:, None] <= _mld[None, :] + 200)
        midlayer_mask &= ~np.isnan(_temp)
        n = midlayer_mask.sum(axis=0)

        tmean = np.nansum(midlayer_mask * _temp, axis=0) / n
        dmean = (midlayer_mask * depth[:, None]).sum(axis=0) / n
        num = np.nansum(((depth[:, None] - dmean[None, :]) * (_temp - tmean[None, :])) * midlayer_mask, axis=0)
        denom = (((depth[:, None] - dmean[None, :]) ** 2) * midlayer_mask).sum(axis=0)

        strat = num / denom

        out_arr = np.zeros_like(mld_da.data)
        out_arr *= np.nan
        out_arr.ravel()[nanmask] = strat

        out_arr1 = np.zeros_like(mld_da.data)
        out_arr1 *= np.nan
        out_arr1.ravel()[nanmask] = dsst

        max_depth = ((~np.isnan(temp_da.data)) * (depth[:, None, None])).max(axis=(0, 1))
        max_depth = ((~np.isnan(temp_da.data)) * (depth[:, None, None])).max(axis=(0, 1))

        outds = xr.Dataset(coords=mld_da.coords)
        outds['mld'] = mld_da
        outds['dsst'] = (mld_da.dims, out_arr1)
        outds['strat'] = (mld_da.dims, out_arr)
        outds['depth'] = (mld_da.dims, max_depth[None, ...])

        month_datasets.append(outds)

        del temp_da
        del temp
        del midlayer_mask
        del latlon_idxs
        del depth_idxs

    month_clim = xr.concat(month_datasets, dim='Time').mean(dim='Time')
    month_clims.append(month_clim)


month_clims = month_clims[-1:] + month_clims
clim = xr.concat(month_clims, dim='month')
clim.to_netcdf(os.path.expanduser("~/ocean_monthly_climatology.nc"))
