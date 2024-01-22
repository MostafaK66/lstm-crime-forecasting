import os
from hyperopt import hp, Trials
import numpy as np
DATA_PATH = os.path.join(os.getcwd(), 'input_data')
PLOTS_PATH = os.path.join(os.getcwd(), 'plots')
TRAINED_PATH = os.path.join(os.getcwd(), 'trained_model')
TIME_H = 7
WINDOW_SHAPE = 21
STEP_SIZE = 1
TEST_SIZE = 0.3
BAYSIAN_ITERATIONS = 5
NUMBER_OF_PATIENCE = 7
MINIMUM_DELTA = 0.001
CAT_VAR = ['month', 'wday', 'day']
# PARAM_GRID = {
#             'lstm_unit': [64, 128],
#             'emb_dim': [16, 32],
#             'lr': [0.0001, 0.001],
#             'epochs': 100,
#             'batch_size': [32, 64],
#             'activation_function':['tanh'],
#             "dropout_ratio":[0.0, 0.3]
#         }

PARAM_GRID = {
            'lstm_unit': [2],
            'emb_dim': [8],
            'lr': [1e-4],
            'epochs': 10,
            'batch_size': [16],
            'activation_function':['relu', 'tanh'],
            "dropout_ratio":[0.0, 0.3]
        }


PARAM_BAYESIAN = {
            'lstm_unit': 8 + hp.randint('lstm_unit', 128),
            'emb_dim': 2 + hp.randint('emb_dim', 16),
            'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
            'epochs': 10,
            'batch_size': 8 + hp.randint('batch_size', 32),
            'dropout_ratio': hp.loguniform('dropout_ratio', np.log(0.1), np.log(0.4))
        }

