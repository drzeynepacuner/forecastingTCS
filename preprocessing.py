#TODO a preprocessing step for padding and combining the TC data for multiple forecasting.
import pandas as pd


class preprocessing():

    def __init__(self):
        super().__init__()

    def monthly_binning(data):
        data.posted_date = pd.to_datetime(data.posted_date)
        data = data.groupby(pd.Grouper(key='posted_date', freq='M')).sum()
        return data
    def outlier_handling(data):
        import numpy as np
        from scipy import stats
       # Drop if first month is 0.
        firstMonthEarnings = int(data.total_earned.iloc[0])
        if firstMonthEarnings == 0:
            data.drop(data.head(1).index, inplace=True)
            #data.reset_index(drop=True, inplace=True)
        print('data average:', np.average(data.total_earned))
        '''
        # Calculate the IQR
        Q1 = data.total_earned.quantile(0.15)
        Q3 = data.total_earned.quantile(0.9)
        IQR = Q3 - Q1

        # Identify and replace outliers with the non-outlier average
        non_outlier_mask = (data.total_earned >= Q1 - 1.5 * IQR) & (data.total_earned <= Q3 + 1.5 * IQR)
        non_outlier_average = data.loc[non_outlier_mask, 'total_earned'].mean()

        data.total_earned = np.where(~non_outlier_mask, non_outlier_average, data.total_earned)
        print('after outlier removal average:', np.average(data.total_earned) )
        return pd.DataFrame(data)

        '''''
        # OLD WAY 
        
        data_lim = data[(data.total_earned < data.total_earned.mean() + 0.5 * data.total_earned.mean()) &
                        (data.total_earned > data.total_earned.mean() - 0.5 * data.total_earned.mean())]

        print('Data average:', np.round(np.average(data.total_earned)), ',', 'Data-lim average',
              np.round(np.average(data_lim.total_earned)))

        data_new = []

        for i in range(0, len(data)):
            if data.total_earned.iloc[i] > np.quantile(data.total_earned, 0.60):
                data_new.append(np.average(data_lim.total_earned))
            elif data.total_earned.iloc[i] < np.quantile(data.total_earned, 0.15):
                data_new.append(np.average(data_lim.total_earned))
            else:
                data_new.append(data.total_earned.iloc[i])

        print('Data-new average:', np.round(np.average(data_new)))
        data_new = pd.DataFrame(data_new)
        data_new.reset_index(inplace=True)

        if data_new.isnull().values.any() == True:
            data_new = data.total_earned
            print('Data not modified.')
        data_new = pd.DataFrame(data_new)
        data_new.reset_index(inplace=True)
        data_new.rename(columns={'index': 'posted_date', 0: 'total_earned'}, inplace=True)
        #data_new.fillna(1, inplace=True)

        return data_new


    def log(data):
        import numpy as np
        return np.log(data.astype(float) + 1)

    def reverse_log(data):
        import numpy as np
        return np.exp(data.astype(float))-1

    def train_test_split(data, term):

        import numpy as np
        split = term
        print('split:', split)
        average = np.average(data.total_earned)
        print(len(data[:split]), len(data[split:]))
        return data[:split], data[split:]

    def to_timeseries(data, average):
        from quickstart import TimeSeries
        data.reset_index(inplace=True)
        series = TimeSeries.from_dataframe(data, "posted_date", "total_earned_diff_log", fillna_value=average)

        return series

    def set_term(data):
        import numpy as np
        historical_sum = sum(data.total_earned.tail(12))

        if historical_sum < 500:
            ticket = 'None'
            term = 0
            print('Not enough earnings.')
        elif 500 < historical_sum < 50000:
            ticket = 'Small'
            term = 12
        elif 50000 < historical_sum < 500000:
            ticket = 'Medium'
            term = 36
        elif 500000 < historical_sum:
            ticket = 'Large'
            term = 48
        return historical_sum, term


    def moving_average(data, window_size):
        import pandas as pd
        return data.rolling(window=window_size).mean()

    def reverse_moving_average(data, data_detrended, window_size):
        return data_detrended + data.rolling(window=window_size).mean()

    def differencing(data):
        return data.diff()

    def reverse_differencing(data, first_value):
        return data.cumsum() + first_value


