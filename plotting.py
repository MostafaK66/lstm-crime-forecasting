import matplotlib.pyplot as plt
import os
import numpy as np

class DataPlotter:
    def __init__(self, plots_path):
        self.plots_path = plots_path
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

    def plot_crimes(self, df):
        plt.figure(figsize=(10, 6))
        df['n_crimes'].plot(title='Number of Crimes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Crimes')

        plt.savefig(os.path.join(self.plots_path, 'crime_plot.png'))
        plt.close()

    def plot_seasonalities(self, month_seasonality, weekday_seasonality):
        plt.figure(figsize=(16, 6))

        plt.subplot(121)
        month_seasonality['median'].plot(ax=plt.gca(), legend=True)
        month_seasonality['q_30'].plot(legend=True)
        month_seasonality['q_70'].plot(legend=True)
        plt.ylabel('Crimes');
        plt.xlabel('Month')

        plt.subplot(122)
        weekday_seasonality['median'].plot(ax=plt.gca(), legend=True)
        weekday_seasonality['q_30'].plot(legend=True)
        weekday_seasonality['q_70'].plot(legend=True)
        plt.ylabel('Crimes');
        plt.xlabel('Weekday')

        plt.tight_layout()

        plots_path = os.path.join(self.plots_path, 'seasonality_plot.png')
        plt.savefig(plots_path)
        plt.close()

    def plot_predictions(self, pred, y_valid, time_h):
        plt.figure(figsize=(16, 6))

        for t_h in range(time_h):
            plt.plot(pred[(time_h - t_h):-(t_h + 1), t_h],
                     c='blue', alpha=1 - 1 / (time_h + 1) * (t_h + 1),
                     label=f"pred day + {t_h + 1}")

        plt.plot(y_valid[time_h:, 0], c='red', alpha=0.6, label='true')

        plt.ylabel('daily crimes')
        plt.xlabel('time')
        plt.legend()

        plots_path = os.path.join(self.plots_path, 'predictions.png')
        plt.savefig(plots_path)
        plt.close()

        np.set_printoptions(False)

    def plot_mse_comparison(self, mse_model, mse_baseline):
        plt.figure(figsize=(14, 5))

        plt.bar(np.arange(len(mse_model)) - 0.15, list(mse_model.values()), alpha=0.5, width=0.3, label='Seq2Seq')
        plt.bar(np.arange(len(mse_baseline)) + 0.15, list(mse_baseline.values()), alpha=0.5, width=0.3,
                label='Baseline')

        plt.xticks(range(len(mse_baseline)), list(mse_baseline.keys()))

        plt.ylabel('MSE')
        plt.legend()

        plots_path = os.path.join(self.plots_path, 'mse_comparison.png')
        plt.savefig(plots_path)
        plt.close()

    def plot_residuals(self, data):
        plt.figure(figsize=(16, 6))

        plt.subplot(121)
        data.plot(ax=plt.gca(), alpha=0.5)
        plt.scatter(data[data.resample('M').apply(lambda x: x.idxmax()).values].index,
                    data[data.resample('M').apply(lambda x: x.idxmax()).values].values, c='red')
        plt.ylabel('residuals')

        plt.subplot(122)
        monthly_max = data.resample('M').max()
        monthly_max.plot(ax=plt.gca(), c='red')
        plt.ylabel('residuals')

        plot_path = os.path.join(self.plots_path, 'residuals_plot.png')
        plt.savefig(plot_path)
        plt.close()




