import netCDF4

import logging
import sys
import os

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

BINARY_THRESH = 0.15


class DataGen:
    def __init__(self, ice_var='sivol'):
        self.scaler = StandardScaler()
        self.X_vars = None
        self.Y_vars = None
        self.ice_var = ice_var

    def create_landmask_from_nans(self, ds, var_='zos'):
        self.landmask = np.logical_not(np.isnan(ds[var_].isel(time=0)))

    @staticmethod
    def create_timesteps(arr, num_timesteps=3):
        timesteps = [arr[: -(num_timesteps - 1)]]

        for i in range(1, num_timesteps - 1):
            timesteps.append(arr[i : -((num_timesteps - 1) - i)])

        timesteps.append(arr[(num_timesteps - 1) :])
        return np.array(timesteps)

    @staticmethod
    def get_add(ds, start_day=244):
        """Calculate ADD. Defaults to using Sept. 1st as start of climatic year (244)"""
        ds["climate_year"] = np.cumsum(
            ds.time.dt.dayofyear == start_day
        )  # Temporarily add variable denoting the meteorological year
        ds["add"] = ds.groupby("climate_year").apply(
            lambda x: (x.t2m - 273.15).cumsum(axis=0)
        )  # Compute ADD using cumsum()
        ds = ds.drop_vars("climate_year")
        return ds

    def get_data(
        self,
        path="/home/zgoussea/scratch/era5_hb_daily.zarr",
        ds=None,
        add_add=True,
        X_vars=None,
        Y_vars=['siconc']
        ):

        if ds is None:
            ds = xr.open_zarr(path)
        
        logging.debug('Read Zarr store.')

        # Calculate ADD
        if add_add:
            ds = self.get_add(ds)

        logging.debug('Calculated accumulated degree-days (ADD)')

        self.Y_vars = Y_vars

        # If no input variables specified, use all except the output variables
        if X_vars is None:
            logging.info('No input variables specified, using all available')
            
            # Re-order dataset so predictand variables are first 
            # This allows subsequent functions to assume the first N variables are the 
            # predictands (once it is transformed to unnamed numpy arrays)
            self.X_vars = list(ds.data_vars)
            for var_ in Y_vars:
                self.X_vars.remove(var_)
            self.X_vars = self.Y_vars + self.X_vars
        else:
            self.X_vars = X_vars

        if add_add:
            if 'add' not in self.X_vars:
                self.X_vars.append('add')

        assert np.all([var_ in ds.data_vars for var_ in self.X_vars])
        assert np.all([var_ in ds.data_vars for var_ in self.Y_vars])
        assert len(self.X_vars) > 0
        assert len(self.Y_vars) > 0
        # Ensure predicted variable is in the full set of variables
        assert np.all([var_ in self.X_vars for var_ in self.Y_vars]), f'Not all of y variables ({self.Y_vars}) are in x variables ({self.X_vars})'

        logging.info(f"Predictor variable(s): ({len(self.X_vars)}) {self.X_vars}")
        logging.info(f"Predictand variable(s): ({len(self.Y_vars)}) {self.Y_vars}")

        ds = ds[self.X_vars]

        ds = ds.assign_coords({"month": ds.time.dt.month})
        self.ds = ds
        return ds

    def get_mean_and_variance(self, ds, cache=True):
        cached_fn = f'{ds.latitude.values.min():.1f}-{ds.latitude.values.max():.1f}-' + \
                    f'{ds.longitude.values.min():.1f}-{ds.longitude.values.max():.1f}'
        if cache:
            try:
                u, std = self.read_mean_and_variance(cached_fn)
            except FileNotFoundError:
                logging.debug('Asked for cached mean/std values but the file did not exist. Resorting to calculating.')
                u, std = self.calculate_mean_and_variance(ds)
                u.to_netcdf(f'cache/mean_{cached_fn}.nc')
                std.to_netcdf(f'cache/std_{cached_fn}.nc')

        else:
            u, std = self.calculate_mean_and_variance(ds)

        for var_ in u:
            assert not np.isnan(u[var_]), f'Encountered NaN in means! ({var_})'
            assert not np.isnan(std[var_]), f'Encountered NaN in standard deviations! ({var_})'

        self.u, self.std = u, std
        return u, std

    def normalize_xarray(self, ds, cache=True):
        u, std = self.get_mean_and_variance(ds, cache=cache)
        return (ds - u) / std

    def calculate_mean_and_variance(self, ds, n_sample=None):
        if n_sample is not None:
            ds = ds.isel(time=np.random.choice(ds.time.size, size=n_sample, replace=False))
            
        u, std = ds.mean(skipna=True), ds.std(skipna=True)
        return u, std 

    def read_mean_and_variance(self, fn):
        if not os.path.exists(f'cache/mean_{fn}.nc') or not os.path.exists(f'cache/std_{fn}.nc'):
            raise FileNotFoundError
            
        u = xr.open_dataset(f'cache/mean_{fn}.nc').load()
        std = xr.open_dataset(f'cache/std_{fn}.nc').load()
        
        return u, std 

    @staticmethod
    def get_3_month_window(center_month):
        if center_month == 1:
            return (12, 1, 2)
        elif center_month == 12:
            return (11, 12, 1)
        else:
            return (center_month - 1, center_month, center_month + 1)

    @staticmethod
    def index_by_months(ds, months):
        ds_grouped = ds.groupby('time.month').groups
        time_idx = [ds_grouped[month] for month in months]
        time_idx = [item for sublist in time_idx for item in sublist]
        return ds.isel(time=time_idx)

    def data_split(self, ds, num_training_years=3, month=1):

        logging.debug('Splitting data')

        # Get +-1 months
        months = self.get_3_month_window(month)

        # Get all data for months +-1 the desired month
        logging.debug('Retrieving relevant months from the dataset')
        ds = self.index_by_months(ds, months)  # ds.where(ds.month.isin(months), drop=True)
        logging.debug('Retrieved relevant months from the dataset')

        # Add year designator
        # Done in a silly way by finding the large breaks in the time dimension (i.e. when going from one year to the next) and
        # using a cumsum to add up all those breaks TODO: do this in a less silly way
        nominal_years_array = np.append(
            0,
            np.cumsum(
                (ds.time.values[1:] - ds.time.values[:-1]).view(int) > (24 * 60 * 60 * 1e9)
            ),
        )
        nominal_years_array = xr.DataArray(
            nominal_years_array, dims=["time"], coords={"time": ds.time}
        )
        # ds = ds.assign_coords({"nominal_year": nominal_years_array})
        # print(nominal_years_array)
        ds['nominal_years'] = list(nominal_years_array)
        # ds.isel(time=slice(0, 100)).to_zarr('test.zarr')
        # ds = ds.set_coords('nominal_years').set_xindex('nominal_years')
        
        # print(ds)
        # # ds = ds.set_coords('nominal_years').set_xindex('nominal_years')
        # ds = ds.set_coords('nominal_years')
        # print(ds)
        # ds = ds.set_xindex('nominal_years')
        # print(ds)

        logging.debug('Added nominal year to dataset')

        # Loop over every year and yield the split
        nominal_years = sorted(np.unique(nominal_years_array))

        # For the January model we have to remove the first year, since we do not have data for Dec of the previous year
        # Same goes for Feburuary because the dataset starts at February, for some reason.
        if (month == 1) or (month == 2):
            nominal_years = nominal_years[1:]

        # First N years for initial training
        training_years = nominal_years[:num_training_years]  # First N years
        test_year = nominal_years[num_training_years]  # Next year for testing
        valid_year = nominal_years[num_training_years + 1]  # Next next year for validation

        ds_train = ds.sel(nominal_years=slice(training_years[0], training_years[-1]))
        ds_test = ds.sel(nominal_years=test_year)
        ds_valid = ds.sel(nominal_years=valid_year)

        # ds_train = ds_month.where(ds_month.nominal_year.isin(training_years), drop=True)
        # ds_test = ds.where(ds.nominal_year == test_year, drop=True)
        # ds_valid = ds.where(ds.nominal_year == valid_year, drop=True)
        ds_valid = ds_valid.where(ds_valid.time.dt.month == month, drop=True)
        logging.debug('Split dataset into train/test/val for the initial training block')
        yield ds_train, ds_test, ds_valid

        # Get train / test years for "fine tuning" years for next iteration
        for year in nominal_years[num_training_years:-3]:
            training_year = nominal_years[year]  # Year 1 for training
            test_year = nominal_years[year + 1]  # Year 2 for testing
            valid_year = nominal_years[year + 2]  # Year 3 for validation

            ds_train = ds.where(
                ds.nominal_years.isin(training_year), drop=True
            )
            ds_test = ds.where(ds.nominal_years == test_year, drop=True)
            ds_valid = ds.where(ds.nominal_years == valid_year, drop=True)
            ds_valid = ds_valid.where(ds_valid.time.dt.month == month, drop=True)
            logging.debug('Split dataset into train/test/val for annual fine-tuning')
            yield ds_train, ds_test, ds_valid

    def update_scaler(self, arr):
        """
        Unsure if partial_fit() is most appropriate. Distribution changes (due to changing climate), so maybe best to
        re-fit from scratch each time?
        """
        self.scaler = self.scaler.partial_fit(arr.reshape(-1, np.prod(arr.shape[1:])).T)

    def normalize(self, arr):
        return self.scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

    def get_transformed_value(self, var_index, value):
        """Get the transformed equivalent of a value for the already-fit scaler."""
        input_vector = np.zeros(shape=(1, self.scaler.n_features_in_))
        input_vector[0][var_index] = value
        trans_vector = self.scaler.transform(input_vector)
        return trans_vector[0][var_index]

    def split_xy(self, arr, num_timesteps_predict, num_timesteps_input, split_index):
        # Split x and y temporally
        Y = arr[:, :, :, :, -(num_timesteps_predict):]
        X = arr[:, :, :, :, :num_timesteps_input]

        # Extract Y variables at the specified index
        # TODO: Extracting SIC as the target variable this late creates a larger array than is necessary. Try to find a
        # way to do the splitting into X/Y earlier to avoid this.
        Y = Y[:split_index]

        # Reshape
        Y = np.transpose(Y, [1, 4, 2, 3, 0])
        X = np.transpose(X, [1, 4, 2, 3, 0])
        return X, Y

    def get_generator(
        self,
        ds,
        month,
        num_timesteps_input=3,
        num_timesteps_predict=30,
        num_training_years=3,
        binary_sic=True,
        valid_only=False,
        normalize=True,
    ):

        if normalize:
            ds = self.normalize_xarray(ds, cache=True)
            
        # logging.debug('Normalized the dataset')

        # Create expanded dataset (add timesteps)
        # TODO: This multiplies the size of the dataset by the number of input timesteps / output timesteps. 
        # There should be a way to do this without duplicating any data. This is what xarray is meant to do 
        # (use pointers and lazy loading) but it doesn't look like it's actually doing that. 
        # 
        # It can probably done fairly easily by keeping track of time indices, and using sel()
        ds_timesteps = ds.rolling(
            time=num_timesteps_input + num_timesteps_predict
        ).construct("timesteps")

        logging.debug('Constructed timesteps')

        # Remove first num_timesteps_input timesteps and assign the launch date
        # dates to be the de-facto dates for each timestep.
        launch_dates = ds_timesteps.time[num_timesteps_input:-num_timesteps_predict]
        ds_timesteps = ds_timesteps.isel(
            time=slice(num_timesteps_input + num_timesteps_predict, None)
        )
        ds_timesteps = ds_timesteps.assign_coords(time=launch_dates)

        for ds_train, ds_test, ds_valid in self.data_split(ds_timesteps, num_training_years, month):
            # if normalize:
            #     logging.debug('Normalizing the datasets')
            #     ds_train = self.normalize_xarray(ds_train, cache=False)
            #     ds_test = self.normalize_xarray(ds_test, cache=False)
            #     ds_valid = self.normalize_xarray(ds_valid, cache=False)

            #     logging.debug('Normalized the datasets')

            # Save dates
            dates_train, dates_test, dates_valid = (
                ds.time for ds in [ds_train, ds_test, ds_valid]
            )

            # Update the normalization scaler with just the first timestep of the train array
            # self.update_scaler(np.array(ds_train.isel(timesteps=0).to_array()))

            # Since this function can also be used to only get validation data (for evaluating results),
            # we first process the validation data only, then add the train/test if it is desired. Maybe not
            # the most intuitive way of doing this.

            logging.debug(f'Variable order: {list(ds_valid.data_vars)}')
            ice_index = list(ds_valid.data_vars).index(self.ice_var)
            
            # Convert to numpy & replace NaNs with 0s (ensure ice NaNs are land and not 0s)
            valid_array = np.array(ds_valid.to_array())
            valid_array[ice_index][np.isnan(valid_array[ice_index])] = (0 - self.u[self.ice_var]) / self.std[self.ice_var]
            valid_array = np.nan_to_num(valid_array)

            valid_X, valid_Y = self.split_xy(
                valid_array, num_timesteps_predict, num_timesteps_input, split_index=len(self.Y_vars)
            )

            # Normalize X only
            # valid_X = self.normalize(valid_X)

            # If we want binary ice off / on instead of SIC
            if binary_sic:
                valid_Y[0] = valid_Y[0] > BINARY_THRESH

            # Add train and test data to the returned dictionary
            data = dict(
                dates_valid=dates_valid,
                valid_X=valid_X,
                valid_Y=valid_Y,
            )

            if not valid_only:
                # Convert to numpy & replace NaNs with 0s
                train_array = np.array(ds_train.to_array())
                train_array[ice_index][np.isnan(train_array[ice_index])] = (0 - self.u[self.ice_var]) / self.std[self.ice_var]
                train_array = np.nan_to_num(train_array)
                
                test_array = np.array(ds_test.to_array())
                test_array[ice_index][np.isnan(test_array[ice_index])] = (0 - self.u[self.ice_var]) / self.std[self.ice_var]
                test_array = np.nan_to_num(test_array)

                train_X, train_Y = self.split_xy(
                    train_array, num_timesteps_predict, num_timesteps_input, split_index=len(self.Y_vars)
                )
                test_X, test_Y = self.split_xy(
                    test_array, num_timesteps_predict, num_timesteps_input, split_index=len(self.Y_vars)
                )

                # Normalize X only
                # train_X = self.normalize(train_X)
                # test_X = self.normalize(test_X)

                # If we want binary ice off / on instead of SIC
                if binary_sic:
                    train_Y[..., 0] = train_Y[..., 0] > BINARY_THRESH
                    test_Y[..., 0] = test_Y[..., 0] > BINARY_THRESH

                # Add train and test data to the returned dictionary
                data = {
                    **data,
                    **dict(
                        dates_train=dates_train,
                        train_X=train_X,
                        train_Y=train_Y,
                        dates_test=dates_test,
                        test_X=test_X,
                        test_Y=test_Y,
                        dates_valid=dates_valid,
                    ),
                }

                logging.info(
                    f"""Generated dataset:
                    \tTraining: {dates_train[0].values} to {dates_train[-1].values}
                    \tTest: {dates_test[0].values} to {dates_test[-1].values}
                    \tValidation: {dates_valid[0].values} to {dates_valid[-1].values}"""
                )
            else:
                logging.info(
                    f"""Generated dataset:
                    \tValidation: {dates_valid[0].values} to {dates_valid[-1].values}"""
                )

            yield data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_gen = DataGen()

    ds = data_gen.get_data("/home/zgoussea/scratch/era5_hb_daily.zarr")
    landmask = data_gen.get_landmask(ds)
    print(landmask.shape)
    plt.figure()
    plt.imshow(landmask)
    plt.savefig(f"figs/landmask.png")

    month = 1
    datasets = data_gen.get_generator(
        ds,
        month,
        num_timesteps_input=3,
        num_timesteps_predict=5,
        predict_only_sic=True,
        num_training_years=1,
        binary=True,
    )

    for data in datasets:
        for y in ["train_Y", "test_Y", "valid_Y"]:
            plt.figure()
            plt.imshow(data[y][-20, 0, :, :, 0])
            plt.savefig(f"figs/{y}_2.png")
        break
    