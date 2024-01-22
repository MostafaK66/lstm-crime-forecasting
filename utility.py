import os
import pandas as pd
import numpy as np
from scaler import Scaler3D
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def read_data(self, file_name):
        file_path = os.path.join(self.data_path, file_name)
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
        df = pd.DataFrame({"n_crimes": df.Date.value_counts().sort_index()})
        df["month"] = df.index.month
        df["wday"] = df.index.dayofweek
        df["day"] = df.index.day
        return df

    def calculate_seasonality(self, df):
        month_seasonality = df.n_crimes.groupby(df.index.month).agg(['median', lambda x: x.quantile(0.3), lambda x: x.quantile(0.7)])
        month_seasonality.columns = ['median', 'q_30', 'q_70']

        weekday_seasonality = df.n_crimes.groupby(df.index.weekday).agg(['median', lambda x: x.quantile(0.3), lambda x: x.quantile(0.7)])
        weekday_seasonality.columns = ['median', 'q_30', 'q_70']

        return month_seasonality, weekday_seasonality

    def create_windows(self, data, window_shape, step, start_id=None, end_id=None):
        data = np.asarray(data)
        data = data.reshape(-1, 1) if np.prod(data.shape) == max(data.shape) else data

        start_id = 0 if start_id is None else start_id
        end_id = data.shape[0] if end_id is None else end_id

        data = data[int(start_id):int(end_id), :]
        window_shape = (int(window_shape), data.shape[-1])
        step = (int(step),) * data.ndim
        slices = tuple(slice(None, None, st) for st in step)
        indexing_strides = data[slices].strides
        win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1

        new_shape = tuple(list(win_indices_shape) + list(window_shape))
        strides = tuple(list(indexing_strides) + list(data.strides))

        window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)

        return np.squeeze(window_data, 1)

    def prepare_seq(self, num_X, cat1_X, cat2_X, cat_var, scaler=None):
        if scaler is not None:
            num_X = scaler.transform(num_X)

        inp_dict = {}
        inp_dict['inp_num'] = num_X
        for i, c in enumerate(cat_var):
            inp_dict[f"inp_{c}_cat1"] = cat1_X[:, :, i]
            inp_dict[f"inp_{c}_cat2"] = cat2_X[:, :, i]

        return inp_dict

    def split_data_indices(self, df_length, window_shape, time_h, test_size):
        indices = np.arange(df_length - window_shape - time_h + 1)
        _id_train, _id_valid = train_test_split(indices, test_size=test_size, shuffle=False)
        return _id_train, _id_valid

    def split_windows(self, window_data, _id_train, _id_valid):
        return window_data[_id_train], window_data[_id_valid]

    def scale_data(self, _num_X_train, _target_train):
        X_scaler = Scaler3D().fit(_num_X_train)
        y_scaler = Scaler3D().fit(_target_train)
        return X_scaler, y_scaler

    def prepare_train_test_data(self, df, time_h, window_shape, step, test_size, cat_var):
        target = ['n_crimes']

        _id_train, _id_valid = self.split_data_indices(len(df), window_shape, time_h, test_size)

        _num_X = self.create_windows(df[target], window_shape=window_shape, step=step, end_id=-time_h)
        _cat1_X = self.create_windows(df[cat_var], window_shape=window_shape, step=step, end_id=-time_h)
        _cat2_X = self.create_windows(df[cat_var], window_shape=time_h, step=step, start_id=window_shape)
        _target = self.create_windows(df[target], window_shape=time_h, step=step, start_id=window_shape)

        _num_X_train, _num_X_valid = self.split_windows(_num_X, _id_train, _id_valid)
        _cat1_X_train, _cat1_X_valid = self.split_windows(_cat1_X, _id_train, _id_valid)
        _cat2_X_train, _cat2_X_valid = self.split_windows(_cat2_X, _id_train, _id_valid)
        _target_train, _target_valid = self.split_windows(_target, _id_train, _id_valid)

        X_scaler, y_scaler = self.scale_data(_num_X_train, _target_train)

        X_train = self.prepare_seq(_num_X_train, _cat1_X_train, _cat2_X_train, cat_var, scaler=X_scaler)
        y_train = y_scaler.transform(_target_train)
        X_valid = self.prepare_seq(_num_X_valid, _cat1_X_valid, _cat2_X_valid, cat_var, scaler=X_scaler)
        y_valid = y_scaler.transform(_target_valid)

        return X_train, y_train, X_valid, y_valid, X_scaler, y_scaler, _id_train, _id_valid







