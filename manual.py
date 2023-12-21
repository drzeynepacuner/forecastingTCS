from preprocessing import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts.utils.missing_values import fill_missing_values
from darts import TimeSeries
from darts.metrics import mape, mase, smape
from darts.models import AutoARIMA, Theta, LinearRegressionModel, MovingAverageFilter


def quickstart():

    data = pd.read_csv('frvrfriday_total.csv', delimiter=';')

    #data.reset_index(inplace=True)
    #data = data.groupby(['Sale Month']).sum()
    data.reset_index(inplace=True)

    # Get historical sum and average.
    historical_sum = sum(data.total_earned)
    historical_average = np.average(data.total_earned)

# Outlier handling

    data = preprocessing.outlier_handling(data)
    data.total_earned = data.total_earned.rolling(2).mean()
    history = TimeSeries.from_dataframe(data, time_col='posted_date', value_cols=['total_earned'], fill_missing_dates=True, freq=None)




    # Clean NaNs and add a constant?



    history = history + 10
    # Take the logarithm.
    history = history.map(np.log)

    series = fill_missing_values(history)

    # Override term.
    term = 9



    train, val = series.split_after(0.75)

    # Search for the best theta parameter, by trying 50 different values
    thetas = 2 - np.linspace(-10, 10, 50)

    best_mape = float("inf")
    best_theta = 0

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

    print(
        "The MAPE is: {:.2f}, with theta = {}.".format(
            mape(val, pred_best_theta), best_theta
        )
    )

    theta = Theta(best_theta)
    theta.fit(series)
    forecast_theta = theta.predict(term)
    forecast_theta = forecast_theta.map(np.exp) -10

    from darts.models import AutoARIMA
    AutoARIMA = AutoARIMA(start_p=1, start_q=1,
                           max_p=3, max_q=3, m=4,
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
    plt.xlabel('Quarter')
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

    results=[]

    results.append([np.array(pd.DataFrame(forecast_AutoARIMA.values)), np.array(pd.DataFrame(forecast_theta.values))])

    print('results', results)

    results = pd.DataFrame(results)


    results.columns = ['forecast_arima','forecast_theta']

    return results.to_csv('results.csv')





quickstart()