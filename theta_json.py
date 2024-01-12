
from preprocessing import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from darts.metrics import mape, mase, smape
from darts.models import AutoARIMA, Theta, LinearRegressionModel, MovingAverageFilter, ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

# Read the data.
folder_path = '/Users/zeynep.acuner/PycharmProjects/forecasting/BMI-Snapshot-Dec-14-2023/'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file
results_all = []
results = []

for file_name in files:
    # Check if the file is a JSON file
    if file_name.endswith('.json'):
        # Construct the full path to the JSON file
        file_path = os.path.join(folder_path, file_name)

        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        import pandas as pd
        meta = data["meta"]
        data = pd.DataFrame(data['data'])

        if len(data) < 3:
            continue

        if meta['granularity'] == 'quarter':
            term = 9
        elif meta['granularity'] == 'month':
            term = 36
        else:
            print('Granularity neither quarterly nor monthly.')
            print(meta['granularity'])

        # Get historical sum and average.
        historical_sum = sum(data.value)
        historical_average = np.average(data.value)

        # Outlier handling
        data = preprocessing.outlier_handling(data)

        # rolling average
        data.value = data.value.rolling(round(len(data) / 4)).mean()

        # Create time series.
        history = TimeSeries.from_dataframe(data, time_col='time', value_cols=['value'],
                                            fill_missing_dates=True, freq=None)

        # Clean NaNs and add a constant
        history = history + 10

        # Take the logarithm.
        history = history.map(np.log)
        series = fill_missing_values(history)

        # Train-test split.
        train, val = series.split_after(0.75)

        # Search for the best theta parameter, by trying 50 different values
        thetas = 2 - np.linspace(-10, 10, 50)

        best_mape = float("inf")
        best_theta = 0
        if len(train) > 2:
            for theta in thetas:
                model = Theta(theta)
                model.fit(train)
                pred_theta = model.predict(len(val))
                res = mape(val, pred_theta)

                if res < best_mape:
                    best_mape = res
                    best_theta = theta

            best_theta_model = Theta(best_theta)
            best_theta_model.fit(train)
            pred_best_theta = best_theta_model.predict(len(val))

        else:
            best_theta = 0.5

        # Fit the model with best_theta.
        theta = Theta(best_theta)
        theta.fit(series)
        forecast_theta = theta.predict(term)
        forecast_theta = forecast_theta.map(np.exp) - 10

        # Plot.
        historical = series.map(np.exp) - 10
        historical.plot(label='Historical')
        forecast_theta.plot(label='Theta')
        plt.xlabel('Quarter')
        plt.ylabel('Total earned')
        plt.interactive(False)
        # Comment out the below line to remove plotting function.
        # plt.show()

        # Convert to numeric and calculate sum.
        forecast_theta = TimeSeries.pd_dataframe(forecast_theta)
        forecast_sum_theta = sum(forecast_theta.value)
        historical = TimeSeries.pd_dataframe(historical)

        # Prepare the results.
        results = []

        results.append(
            'Granularity:', meta['granularity'], 'Historical sum:', np.round(historical_sum), 'Historical length:',
             len(historical),
             'Historical average:', np.round(historical_average),
             'Theta_sum:', np.round(forecast_sum_theta),
             'Theta average:', np.round(np.average(forecast_theta)))

        print(results)

        results_all.append(results)

results_all = pd.DataFrame(results_all)
results_all.to_csv('results.csv')




