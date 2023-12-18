class metrics():

    def __init__(self):
        super().__init__()

    def tidy(pred):
        import pandas as pd
        from preprocessing import preprocessing
        import numpy as np
        #pred = pred.reset_index()
        pred = pd.DataFrame(pred.to_numpy())
        #val_res = preprocessing.reverse_log(val.total_earned)
        #pred_res = preprocessing.reverse_log(np.array(pred[1], dtype=object))
        pred_res = pred

        return pred_res

    def mae(val, pred):
        # mean absolute error - scale dependent - recommended for assessing accuracy on a single series -
        # unit same with data
        import numpy as np
        return np.round(np.mean(np.abs(val - pred)))

    def mse(val, pred):
        # mean squared error - more attention on outliers - unit not the same with data
        import numpy as np
        return np.mean(np.square(val - pred))

    def rmse(val, pred):
        # root mean squared error - same with mse but unit same with data
        import numpy as np
        return np.round(np.sqrt(np.mean(np.square(val - pred))))

    def mape(val, pred):
        # mean absolute percentage error - scale independent - trouble for near zero values -
        # heavier penalty on negative errors
        import numpy as np
        try:
            return round(np.mean(np.abs((val - pred) / val) * 100))
        except OverflowError:
            print('meh')

    def smape_original(pred, val):
        # symmetric mean absolute percentage error - remedy for mape asymmetry- between 0% and 200%
        # average across all forecasts made for a given horizon
        import numpy as np
        return 1 / len(pred) * np.sum(2 * np.abs(val - pred) / (np.abs(pred) + np.abs(val)) * 100)

    def smape_adjusted(pred, val):
        # median relative absolute error - adjusted SMAPE version to scale metric from 0%-100%
        import numpy as np
        return np.round((1 / pred.size * np.sum(np.abs(val - pred) / (np.abs(pred) + np.abs(val)) * 100)))

    def mdrae(val, pred, bnchmrk):
        # geometric median relative absolute error- median of the difference between the absolute error
        # of our forecast to the absolute error of a benchmark model.
        import numpy as np
        return np.median(np.abs(val - pred) / np.abs(val - bnchmrk))

    def mase(val, pred, pred_train):
        # mean absolute scale error - The MASE is calculated by taking the MAE and dividing it by the MAE of an in-sample
        # (so based on our training data) naive benchmark.
        # Values of MASE greater than 1 indicate that the forecasts are worse,
        # on average, than in-sample one-step forecasts from the naive model (Hyndman and Koehler, 2006).
        ## Naive in-sample Forecast
        import numpy as np
        naive_y_hat = pred_train[:-1]
        naive_y = pred_train[1:]
        ## Calculate MAE (in sample)
        mae_in_sample = np.mean(np.abs(naive_y - naive_y_hat))
        mae = np.mean(np.abs(val - pred))
        return mae / mae_in_sample

    def gmrae(val, pred, bnchmrk):
        # geometric mean relative absolute error
        # >1 benchmark better
        import numpy as np
        abs_scaled_errors = np.abs(val - pred) / np.abs(val - bnchmrk)
        return np.exp(np.mean(np.log(abs_scaled_errors)))