from preprocessing import preprocessing
from model_repo import TSmodels
from res_metrics import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mape, mase, smape
import random

def quickstart():

    data = pd.read_csv('DistroKid_wakeelthomas.csv', delimiter=';')
    data['total_earned'] = data['Team Percentage'] / 100 * data['Earnings (USD)']
    print(sum(data['total_earned']))
    #data.reset_index(inplace=True)
    data = data.groupby(['Sale Month']).sum()
    data.reset_index(inplace=True)

    # Get historical sum and average.
    historical_sum = sum(data.total_earned)
    historical_average = np.average(data.total_earned)

# Outlier handling

    data = preprocessing.outlier_handling(data)
    print(data)
    history = TimeSeries.from_dataframe(data, time_col='posted_date', value_cols=['total_earned'], fill_missing_dates=True, freq=None)


    # Clean NaNs and add a constant?
    from darts.utils.missing_values import fill_missing_values


    history = history + 10
    # Take the logarithm and first differences.
    history = history.map(np.log)

    series = fill_missing_values(history)
    # Override term.
    term = 36


    from darts.models import AutoARIMA, Theta, LinearRegressionModel

    theta = Theta()
    theta.fit(series)
    forecast_theta = theta.predict(term)
    forecast_theta = forecast_theta.map(np.exp) -10


    AutoARIMA = AutoARIMA(start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
    AutoARIMA.fit(series)
    forecast_AutoARIMA = AutoARIMA.predict(term)
    forecast_AutoARIMA = forecast_AutoARIMA.map(np.exp) -10

    historical = series.map(np.exp) - 10

    historical.plot(label='Historical')
    forecast_theta.plot(label='Theta')
    forecast_AutoARIMA.plot(label='AutoArima')
    plt.xlabel('Month')
    plt.ylabel('Total earned')
    plt.interactive(False)
    plt.show()

    forecast_theta = TimeSeries.pd_dataframe(forecast_theta)
    forecast_sum_theta = sum(forecast_theta.total_earned)
    forecast_AutoARIMA = TimeSeries.pd_dataframe(forecast_AutoARIMA)
    forecast_sum_AutoARIMA = sum(forecast_AutoARIMA.total_earned)
    historical = TimeSeries.pd_dataframe(historical)


    print('Term:', 36, 'Historical sum:', np.round(historical_sum), 'Historical length:', len(historical),
          'Historical average:', np.round(historical_average),
          'Theta_sum:', np.round(forecast_sum_theta),
          'Theta average:', np.round(np.average(forecast_theta)),
          'Arima_sum', np.round(forecast_sum_AutoARIMA),
          'Arima average:', np.round(np.average(forecast_AutoARIMA)))

 


    return





quickstart()