import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as mse
import pandas as pd

class ModelValidator:
    def __init__(self, y_scaler, trained_path, model_name):
        self.y_scaler = y_scaler
        self.model_name = model_name
        self.trained_path = trained_path

    def load_model(self):
        model_path = os.path.join(self.trained_path, self.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return load_model(model_path)

    def validate_model(self, X_valid, y_valid):
        best_model = self.load_model()

        pred = np.squeeze(
            self.y_scaler.inverse_transform(
                best_model.predict(X_valid)
            ), -1)

        y_valid_transformed = np.squeeze(self.y_scaler.inverse_transform(y_valid), -1)

        return pred, y_valid_transformed
    def calculate_mse(self, y_valid, pred, time_h):
        mse_model = {}
        mse_baseline = {}

        for t_h in range(time_h):
            _mse = mse(y_valid[1:, t_h], y_valid[:-1, 0])
            mse_baseline[f"day + {t_h + 1}"] = _mse

            _mse = mse(y_valid[:, t_h], pred[:, t_h])
            mse_model[f"day + {t_h + 1}"] = _mse

        return mse_model, mse_baseline

    def compute_residuals(self, df, y_valid, pred, _id_valid, t_h=1):
        resid = np.abs(y_valid[:, t_h] - pred[:, t_h])
        df_residual = pd.Series(resid, index=df.iloc[_id_valid[0]:_id_valid[-1] + 1].index, name='resid')
        return df_residual[:]

