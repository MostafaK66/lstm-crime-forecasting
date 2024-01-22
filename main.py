from utility import DataProcessor
from plotting import DataPlotter
import settings
from model_GS import DeepLearningModelGS
from model_BS import DeepLearningModelBS
from validation import ModelValidator
import time


def main(model_type="BS"):
    start_time = time.time()
    preprocessor = DataProcessor(data_path=settings.DATA_PATH)
    plotter = DataPlotter(plots_path=settings.PLOTS_PATH)
    if model_type == "GS":
        lstm = DeepLearningModelGS(window_shape=settings.WINDOW_SHAPE, cat_var=settings.CAT_VAR, time_h=settings.TIME_H,
                                   train_model=True)
    elif model_type == "BS":
        lstm = DeepLearningModelBS(window_shape=settings.WINDOW_SHAPE, cat_var=settings.CAT_VAR, time_h=settings.TIME_H,
                                   train_model=True)
    else:
        raise ValueError("Invalid model type. Choose 'GS' for Grid Search or 'BS' for Bayesian Search.")

    df = preprocessor.read_data(file_name="crime.csv.zip")
    plotter.plot_crimes(df)
    month_seasonality, weekday_seasonality = preprocessor.calculate_seasonality(df)
    plotter.plot_seasonalities(month_seasonality=month_seasonality, weekday_seasonality=weekday_seasonality)
    X_train, y_train, X_valid, y_valid, X_scaler, y_scaler, _id_train, _id_valid = preprocessor.prepare_train_test_data(
        df=df, time_h=settings.TIME_H, window_shape=settings.WINDOW_SHAPE, step=settings.STEP_SIZE,
        test_size=settings.TEST_SIZE, cat_var=settings.CAT_VAR)

    if lstm.train_model:
        best_model, best_params, best_score = lstm.perform_hyperparameter_tuning(X_train=X_train, y_train=y_train,
                                                                                 X_valid=X_valid, y_valid=y_valid)
        lstm.save_best_model(best_model=best_model, trained_path=settings.TRAINED_PATH,
                             model_name='best_model.h5.keras')
    else:
        best_model, _ = lstm.perform_hyperparameter_tuning(X_train=X_train, y_train=y_train, X_valid=X_valid,
                                                           y_valid=y_valid)

    validator = ModelValidator(y_scaler=y_scaler, trained_path=settings.TRAINED_PATH, model_name='best_model.h5.keras')
    pred, y_valid_transformed = validator.validate_model(X_valid=X_valid, y_valid=y_valid)
    plotter.plot_predictions(pred=pred, y_valid=y_valid_transformed, time_h=settings.TIME_H)
    mse_model, mse_baseline = validator.calculate_mse(y_valid=y_valid_transformed, pred=pred, time_h=settings.TIME_H)
    plotter.plot_mse_comparison(mse_model=mse_model, mse_baseline=mse_baseline)
    df_residual = validator.compute_residuals(df=df, y_valid=y_valid_transformed, pred=pred, _id_valid=_id_valid, t_h=1)
    plotter.plot_residuals(data=df_residual)

    end_time = time.time()
    total_duration = end_time - start_time
    if lstm.train_model:
        lstm.save_best_params(best_params=best_params, best_score=best_score, total_duration=total_duration,
                               trained_path=settings.TRAINED_PATH)





if __name__ == '__main__':
    main()







