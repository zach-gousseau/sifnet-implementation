import numpy as np
import sys
import xarray as xr
from sklearn.preprocessing import StandardScaler
import logging

BINARY_THRESH = 0.15

class DataGen:
    def __init__(self):
        self.scaler = None

    @staticmethod
    def get_landmask(ds):
        return np.logical_not(np.isnan(ds.sic.isel(time=0)))

    @staticmethod
    def create_timesteps(arr, num_timesteps=3):
        timesteps = [arr[:-(num_timesteps - 1)]]
        
        for i in range(1, num_timesteps - 1):
            timesteps.append(arr[i:-((num_timesteps-1)-i)])
                    
        timesteps.append(arr[(num_timesteps - 1):])
        return np.array(timesteps)

    @staticmethod
    def get_add(ds, start_day=244):
        """Calculate ADD. Defaults to using Sept. 1st as start of climatic year (244)"""
        ds['climate_year'] = np.cumsum(ds.time.dt.dayofyear == start_day)  # Temporarily add variable denoting the meteorological year
        ds['add'] = ds.groupby('climate_year').apply(lambda x: (x.t2m - 273.15).cumsum(axis=0))  # Compute ADD using cumsum()
        ds = ds.drop_vars('climate_year')
        return ds

    def get_data(self, path='/home/zgoussea/scratch/era5_hb_daily.zarr'):
        ds = xr.open_zarr(path)
        ds = xr.merge([ds[['siconc']], ds.drop('siconc')])
        ds = ds.rename({'siconc': 'sic'})

        # Only these variables
        vars = ['sic', 'sst', 't2m', 'sshf', 'u10', 'v10']
        ds = ds[vars]

        # Calculate ADD
        ds = self.get_add(ds)

        ds = ds.assign_coords({'month': ds.time.dt.month})
        return ds

    @staticmethod
    def normalize_xarray(ds):
        return (ds - ds.mean()) / ds.std()

    @staticmethod
    def get_3_month_window(center_month):
        if center_month == 1:
            return (12, 1, 2)
        elif center_month == 12:
            return (11, 12, 1)
        else:
            return (center_month - 1, center_month, center_month + 1)

    def data_split(self, ds, num_training_years=3, month=1):
        
        # Get +-1 months
        months = self.get_3_month_window(month)
        
        # Get all data for months +-1 the desired month
        ds_month = ds.where(ds.month.isin(months), drop=True)
        
        # Add year designator
        # Done in a silly way by finding the large breaks in the time dimension (i.e. when going from one year to the next) and 
        # using a cumsum to add up all those breaks
        nominal_years_array = np.append(0, np.cumsum((ds_month.time.values[1:] - ds_month.time.values[:-1]).view(int) > (24*60*60*1e9)))
        nominal_years_array = xr.DataArray(nominal_years_array, dims=['time'], coords={'time': ds_month.time})
        ds_month = ds_month.assign_coords({'nominal_year': nominal_years_array})

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

        ds_train = ds_month.where(ds_month.nominal_year.isin(training_years), drop=True)
        ds_test = ds_month.where(ds_month.nominal_year == test_year, drop=True)
        ds_valid = ds_month.where(ds_month.nominal_year == valid_year, drop=True)
        ds_valid = ds_valid.where(ds_valid.time.dt.month == month, drop=True)
        yield ds_train, ds_test, ds_valid
        
        # Get train / test years for "fine tuning" years for next iteration 
        for year in nominal_years[num_training_years: -3]:
            training_year = nominal_years[year]  # Year 1 for training
            test_year = nominal_years[year + 1]  # Year 2 for testing
            valid_year = nominal_years[year + 2]  # Year 3 for validation

            ds_train = ds_month.where(ds_month.nominal_year.isin(training_year), drop=True)
            ds_test = ds_month.where(ds_month.nominal_year == test_year, drop=True)
            ds_valid = ds_month.where(ds_month.nominal_year == valid_year, drop=True)
            ds_valid = ds_valid.where(ds_valid.time.dt.month == month, drop=True)
            yield ds_train, ds_test, ds_valid
            
    def normalize(self, train_array, test_array, valid_array=None):
        scaler = StandardScaler()
        train_array = scaler.fit_transform(train_array.reshape(-1, np.prod(train_array.shape[1:])).T).T.reshape(train_array.shape)
        test_array = scaler.transform(test_array.reshape(-1, np.prod(test_array.shape[1:])).T).T.reshape(test_array.shape)

        self.scaler = scaler

        if valid_array is not None:
            valid_array = scaler.transform(valid_array.reshape(-1, np.prod(valid_array.shape[1:])).T).T.reshape(valid_array.shape)
            return train_array, test_array, valid_array
        else:
            return train_array, test_array

    def get_transformed_value(self, var_index, value):
        """Get the transformed equivalent of a value for the already-fit scaler."""
        input_vector = np.zeros(shape=(1, self.scaler.n_features_in_))
        input_vector[0][var_index] = value
        trans_vector = self.scaler.transform(input_vector)
        return trans_vector[0][var_index]

    def split_xy(self, arr, num_timesteps_predict, num_timesteps_input):
        # Split x and y temporally
        Y = arr[:, :, :, :, -(num_timesteps_predict):]
        X = arr[:, :, :, :, :num_timesteps_input]

        # Get only SIC from the Y data Assumes SIC is the first variable !!
        # TODO: Extracting SIC as the target variable this late creates a larger array than is necessary. Try to find a 
        # way to do the splitting into X/Y earlier to avoid this. 
        Y = np.expand_dims(Y[0], 0)

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
            binary_SIC=True,
            valid_only=False):

        sic_index = 0

        data_vars = list(ds.data_vars)
        logging.info(f'Read dataset with variables: {data_vars}')
        logging.info(f'Assuming variable {sic_index} ({data_vars[sic_index]}) is the SIC variable')

        # Normalizing the entire dataset first is cheating a little bit since it gives future information to past observations
        # but the effect is likely minimal 
        ds = self.normalize_xarray(ds)

        # Create expanded dataset (add timesteps)
        ds_timesteps = ds.rolling(time=num_timesteps_input + num_timesteps_predict).construct('timesteps')

        # Remove first num_timesteps_input timesteps and assign the launch date
        # dates to be the de-facto dates for each timestep.
        launch_dates = ds_timesteps.time[num_timesteps_input: -num_timesteps_predict]
        ds_timesteps = ds_timesteps.isel(time=slice(num_timesteps_input + num_timesteps_predict, None))
        ds_timesteps = ds_timesteps.assign_coords(time=launch_dates)

        for ds_train, ds_test, ds_valid in self.data_split(ds_timesteps, num_training_years, month):
            
            # Save dates
            dates_train, dates_test, dates_valid = (ds.time for ds in [ds_train, ds_test, ds_valid])

            # Since this function can also be used to only get validation data (for evaluating results), 
            # we first process the validation data only, then add the train/test if it is desired. Maybe not
            # the most intuitive way of doing this. 

            # Convert to numpy & replace NaNs with 0s
            valid_array = np.nan_to_num(np.array(ds_valid.to_array()))

            valid_X, valid_Y = self.split_xy(valid_array, num_timesteps_predict, num_timesteps_input)

            # If we want binary ice off / on instead of SIC
            if binary_SIC:
                valid_Y[0] = valid_Y[0] > BINARY_THRESH

            # Add train and test data to the returned dictionary
            data = dict(
                dates_valid=dates_valid,
                valid_X=valid_X,
                valid_Y=valid_Y,
            )


            if not valid_only:
                # Convert to numpy & replace NaNs with 0s
                train_array = np.nan_to_num(np.array(ds_train.to_array()))
                test_array = np.nan_to_num(np.array(ds_test.to_array()))

                train_X, train_Y = self.split_xy(train_array, num_timesteps_predict, num_timesteps_input)
                test_X, test_Y = self.split_xy(test_array, num_timesteps_predict, num_timesteps_input)

                # If we want binary ice off / on instead of SIC
                if binary_SIC:
                    train_Y[0] = train_Y[0] > BINARY_THRESH
                    test_Y[0] = test_Y[0] > BINARY_THRESH

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
                        )
                    }

                logging.info(
                    f'''Generated dataset:
                    \tTraining: {dates_train[0].values} to {dates_train[-1].values}
                    \tTest: {dates_test[0].values} to {dates_test[-1].values}
                    \tValidation: {dates_valid[0].values} to {dates_valid[-1].values}'''
                )
            else:
                logging.info(
                    f'''Generated dataset:
                    \tValidation: {dates_valid[0].values} to {dates_valid[-1].values}'''
                )

            yield data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_gen = DataGen()

    ds = data_gen.get_data('/home/zgoussea/scratch/era5_hb_daily.zarr')
    landmask = data_gen.get_landmask(ds)
    print(landmask.shape)
    plt.figure()
    plt.imshow(landmask)
    plt.savefig(f'figs/landmask.png')
    
    month = 1
    datasets = data_gen.get_generator(
        ds,
        month,
        num_timesteps_input=3,
        num_timesteps_predict=5,
        predict_only_sic=True,
        num_training_years=1,
        binary=True)

    for data in datasets:
        for y in ['train_Y', 'test_Y', 'valid_Y']:
            plt.figure()
            plt.imshow(data[y][-20, 0, :, :, 0])
            plt.savefig(f'figs/{y}_2.png')
        break