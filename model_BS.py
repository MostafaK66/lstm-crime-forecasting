import os
import json
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import random
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerashypetune import KerasBayesianSearch
import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from hyperopt import hp, Trials


class DeepLearningModelBS:
    def __init__(self, window_shape, cat_var, time_h, train_model):
        self.window_shape = window_shape
        self.cat_var = cat_var
        self.time_h = time_h
        self.param_grid = settings.PARAM_BAYESIAN
        self.beysian_iterations = settings.BAYSIAN_ITERATIONS
        self.number_of_patience = settings.NUMBER_OF_PATIENCE
        self.min_delta = settings.MINIMUM_DELTA
        self.train_model = train_model

    def set_seed(self, seed):
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def lstm_enc_dec(self, param):

        self.set_seed(33)

        inp_num = Input((self.window_shape, 1), name='inp_num')

        embeddings = {
            'month': Embedding(12 + 1, param['emb_dim']),
            'wday': Embedding(6 + 1, param['emb_dim']),
            'day': Embedding(31 + 1, param['emb_dim'])
        }

        inp_cat1, emb_cat = [], []
        for c in self.cat_var:
            _inp_c1 = Input((self.window_shape,), name=f"inp_{c}_cat1")
            emb = embeddings[c](_inp_c1)
            inp_cat1.append(_inp_c1)
            emb_cat.append(emb)

        enc = Concatenate()([inp_num] + emb_cat)
        enc = LSTM(param['lstm_unit'], return_sequences=True)(enc)
        enc = LSTM(param['lstm_unit'], return_sequences=False)(enc)

        inp_cat2, emb_future = [], []
        for c in self.cat_var:
            _inp_c2 = Input((self.time_h,), name=f"inp_{c}_cat2")
            emb = embeddings[c](_inp_c2)
            inp_cat2.append(_inp_c2)
            emb_future.append(emb)

        x = RepeatVector(self.time_h)(enc)
        dec = Concatenate()([x] + emb_future)
        dec = LSTM(param['lstm_unit'], return_sequences=True)(dec)
        dec = LSTM(param['lstm_unit'], return_sequences=True)(dec)

        output = TimeDistributed(Dense(1))(dec)

        model = Model([inp_num] + inp_cat1 + inp_cat2, output)
        model.compile(optimizer=Adam(learning_rate=param['lr'], clipvalue=1.0), loss='mse')

        return model

    def load_model(self, trained_path, model_name):
        model_path = os.path.join(trained_path, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return load_model(model_path)

    def perform_hyperparameter_tuning(self, X_train, y_train, X_valid, y_valid):

        if not self.train_model:
            print("Loading previously trained model.")
            return self.load_model(settings.TRAINED_PATH, 'best_model.h5.keras'), None

        es = EarlyStopping(patience=self.number_of_patience, verbose=1, min_delta=self.min_delta, monitor='val_loss', mode='auto',
                           restore_best_weights=True)

        hypermodel = self.lstm_enc_dec
        kbs = KerasBayesianSearch(hypermodel, self.param_grid, n_iter=self.beysian_iterations, sampling_seed=123, monitor='val_loss', greater_is_better=False, tuner_verbose=2)


        kbs.search(X_train, y_train, trials=Trials(), validation_data=(X_valid, y_valid), callbacks=[es])

        best_params = kbs.best_params
        best_model = kbs.best_model
        best_score = kbs.best_score

        return best_model, best_params, best_score

    def save_best_model(self, best_model, trained_path, model_name):
        if not model_name.endswith('.keras'):
            model_name += '.keras'

        if not os.path.exists(trained_path):
            os.makedirs(trained_path)

        model_path = os.path.join(trained_path, model_name)

        best_model.save(model_path)
        print(f"Model saved to {model_path}")

    def save_best_params(self, best_params, best_score, trained_path, total_duration, parameter_tuning="BS"):
        params_path = os.path.join(trained_path, 'best_params_and_score.txt')

        new_data = f"Best Params: {best_params}\nBest Score: {best_score}\nTotal Duration: {total_duration} seconds\nParameter Tuning: {parameter_tuning}\n\n"

        with open(params_path, 'a') as file:
            file.write(new_data)

        print(f"Parameters and score saved to {params_path}")







