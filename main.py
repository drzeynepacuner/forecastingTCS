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
    results = []
    #members = np.unique(data.member_token)
    results = []
    #for member in members:

    # monthly binning, establish term
    data = pd.read_csv('bq-results-20231122-143519-1700663741796.csv')

    for member in np.unique(data.member_token):
        print('Member token:', member)
        data_of_member = pd.DataFrame(data[data.member_token == member])
        data_of_member = preprocessing.monthly_binning(data_of_member)
        data_of_member.reset_index(inplace=True)
        historical_sum, term = preprocessing.set_term(data_of_member)
        if term == 0:
            continue
        print('Term:', term, ',', 'Historical sum:', np.round(historical_sum), 'Historical length:',  len(data_of_member))

        # Outlier handling
        data_of_member = pd.DataFrame(preprocessing.outlier_handling(data_of_member))
        # Set term.
        historical_sum, term = preprocessing.set_term(data_of_member)
        # Convert to timeseries.
        series = TimeSeries.from_dataframe(data_of_member, time_col='posted_date', value_cols=['total_earned'])

        # Clean NaNs and add a constant?
        from darts.utils.missing_values import fill_missing_values


        series = series + 10
        # Take the logarithm and first differences.
        series = series.map(np.log)

        series = fill_missing_values(series)

        # Train test split.
        train, val = series[:(term)], series[(term):]

        # Manage term as validation.
        if len(train) == 0 or len(val) == 0 or len(val) == 1 or term ==0:
            print('Not enough data.')
            continue
        else:
           # if term < 12:
           #     print('Term smaller than 12 months. To be implemented.')
           #     continue
            if (len(val) < term):
                term = len(val)
            if len(val) > term:
                train, val = series[:(len(data_of_member) - term)], series[(len(data_of_member) - term):]

        #train.plot()
        #val.plot()
        #plt.show()
        #print('series', series.values)

        from darts.models import AutoARIMA, Theta, LinearRegressionModel

        def eval_model(model):

            model.fit(train)
            forecast = model.predict(len(val))

            #train.plot()
            #forecast.plot()
            #val.plot()
            #plt.title(model)
            #plt.show()

            forecast = forecast.map(np.exp) -10

            validation = val.map(np.exp) -10

            training = train.map(np.exp)
            '''
            training.plot()
            forecast.plot()
            validation.plot()
            plt.title(model)
            plt.show()
             '''

            mp = smape(validation, forecast)
            forecast = TimeSeries.pd_dataframe(forecast)
            validation = TimeSeries.pd_dataframe(validation)
            valsum = sum(validation.total_earned)
            forecast_sum = sum(forecast.total_earned)
            print('val', validation)
            print('forecast', forecast)
            print('valsum', valsum)
            print('forecast_sum', forecast_sum)
            return mp, forecast_sum, valsum


        mape_arima, forecast_arima, valsum = eval_model(AutoARIMA())
        mape_theta, forecast_theta, valsum = eval_model(Theta())
        mape_LR, forecast_LR, valsum = eval_model(LinearRegressionModel(lags=2))


        def percent_error(valsum, forecast_sum):
            return ((valsum- forecast_sum)/valsum)*100

        if mape_arima < mape_theta and mape_arima < mape_LR:
            best_model = 'arima'
        elif mape_theta < mape_arima and mape_theta < mape_LR:
            best_model = 'theta'
        else:
            best_model = 'LR'

        results.append([member, best_model, term, valsum,
                        percent_error(valsum,forecast_arima), mape_arima, forecast_arima,
                        percent_error(valsum,forecast_theta), mape_theta, forecast_theta,
                        percent_error(valsum,forecast_LR), mape_LR, forecast_LR])
        print('results', results)
    results = pd.DataFrame(results)


    results.columns = ['member', 'best_model', 'term', 'valsum',
                       'percent_err_arima', 'mape_arima', 'forecast_arima',
                       'percent_err_theta', 'mape_theta', 'forecast_theta',
                       'percent_err_LR','mape_LR', 'forecast_LR']

    return results.to_csv('results.csv')


quickstart()