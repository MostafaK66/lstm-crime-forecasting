"""Optional TensorFlow/KerasTuner integration.

This module is isolated so the deterministic application core can be tested without a
GPU, TensorFlow installation, model download, or large dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crime_forecasting.backend import ModelResult
from crime_forecasting.config import ModelConfig, OutputConfig
from crime_forecasting.errors import ModelBackendError
from crime_forecasting.windows import PreparedData


def run_tensorflow(
    data: PreparedData,
    model_config: ModelConfig,
    output: OutputConfig,
) -> ModelResult:
    """Train/tune or load an encoder-decoder LSTM and forecast validation data."""
    try:
        import keras_tuner as kt
        import tensorflow as tf
    except ImportError as error:
        raise ModelBackendError(
            "TensorFlow support is not installed; run "
            "'python -m pip install -e .[tensorflow]'"
        ) from error

    tf.keras.utils.set_random_seed(model_config.random_seed)
    output.model_directory.mkdir(parents=True, exist_ok=True)
    destination = output.model_directory / output.model_file
    if model_config.train:
        model, parameters, score = _tune(
            tf=tf,
            kt=kt,
            data=data,
            config=model_config,
            tuner_directory=output.model_directory / "tuner",
        )
        model.save(destination)
    else:
        if not destination.is_file():
            raise ModelBackendError(f"Configured model does not exist: {destination}")
        model = tf.keras.models.load_model(destination)
        parameters = {}
        score = None
    scaled = np.asarray(
        model.predict(data.validation.inputs, verbose=0), dtype=np.float64
    )
    predictions = data.target_scaler.inverse_transform(scaled)[:, :, 0]
    return ModelResult(
        predictions=predictions,
        parameters=parameters,
        validation_loss=score,
        model_path=destination,
    )


def _tune(
    *,
    tf: Any,
    kt: Any,
    data: PreparedData,
    config: ModelConfig,
    tuner_directory: Path,
) -> tuple[Any, dict[str, object], float | None]:
    history = data.train.inputs["numeric_history"].shape[1]
    horizon = data.train.targets.shape[1]

    def build(hp: Any) -> Any:
        units = hp.Choice("lstm_units", [32, 64])
        embedding = hp.Choice("embedding_dimension", [8, 16])
        activation = hp.Choice("activation", ["relu", "tanh"])
        dropout = hp.Choice("dropout", [0.0, 0.3])
        numeric = tf.keras.Input((history, 1), name="numeric_history")
        past_month = tf.keras.Input((history,), dtype="int32", name="past_month")
        past_weekday = tf.keras.Input((history,), dtype="int32", name="past_weekday")
        past_day = tf.keras.Input((history,), dtype="int32", name="past_day")
        future_month = tf.keras.Input((horizon,), dtype="int32", name="future_month")
        future_weekday = tf.keras.Input((horizon,), dtype="int32", name="future_weekday")
        future_day = tf.keras.Input((horizon,), dtype="int32", name="future_day")
        month_embedding = tf.keras.layers.Embedding(13, embedding)
        weekday_embedding = tf.keras.layers.Embedding(7, embedding)
        day_embedding = tf.keras.layers.Embedding(32, embedding)
        past_features = tf.keras.layers.Concatenate()(
            [
                numeric,
                month_embedding(past_month),
                weekday_embedding(past_weekday),
                day_embedding(past_day),
            ]
        )
        future_features = tf.keras.layers.Concatenate()(
            [
                month_embedding(future_month),
                weekday_embedding(future_weekday),
                day_embedding(future_day),
            ]
        )
        encoded = tf.keras.layers.LSTM(units, activation=activation, dropout=dropout)(
            past_features
        )
        repeated = tf.keras.layers.RepeatVector(horizon)(encoded)
        decoder_input = tf.keras.layers.Concatenate()([repeated, future_features])
        decoded = tf.keras.layers.LSTM(
            units, activation=activation, dropout=dropout, return_sequences=True
        )(decoder_input)
        forecast = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(decoded)
        network = tf.keras.Model(
            inputs=[
                numeric,
                past_month,
                past_weekday,
                past_day,
                future_month,
                future_weekday,
                future_day,
            ],
            outputs=forecast,
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [0.0001, 0.001]), clipvalue=1.0
        )
        network.compile(optimizer=optimizer, loss="mse")
        return network

    tuner_class = kt.GridSearch if config.strategy == "grid" else kt.BayesianOptimization
    tuner = tuner_class(
        build,
        objective="val_loss",
        max_trials=config.max_trials,
        seed=config.random_seed,
        directory=tuner_directory,
        project_name="crime_forecast",
        overwrite=True,
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        min_delta=config.minimum_delta,
        restore_best_weights=True,
    )
    tuner.search(
        data.train.inputs,
        data.train.targets,
        validation_data=(data.validation.inputs, data.validation.targets),
        epochs=config.epochs,
        callbacks=[callback],
        verbose=1,
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_parameters: dict[str, object] = dict(tuner.get_best_hyperparameters(1)[0].values)
    trial = tuner.oracle.get_best_trials(1)[0]
    score = None if trial.score is None else float(trial.score)
    return best_model, best_parameters, score
