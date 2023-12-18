class TSmodels:

    def __init__(self):
        super().__init__()

    def Theta(train, term):
        from quickstart.models import Theta
        theta_model = Theta()
        theta_model.fit(train)
        pred_theta = theta_model.predict(term).pd_dataframe().reset_index()
        return pred_theta

    def NaiveDrift(train, term):
        from quickstart.models import NaiveDrift
        naivedrift_model = NaiveDrift()
        naivedrift_model.fit(train)
        pred_naivedrift = naivedrift_model.predict(term).pd_dataframe().reset_index()
        return pred_naivedrift

    def NaiveEnsembleModel(train, term, k, l):
        from quickstart.models import NaiveEnsembleModel, NaiveSeasonal, LinearRegressionModel
        naiveensemble_model = NaiveEnsembleModel([NaiveSeasonal(K=k), LinearRegressionModel(lags=l)])
        naiveensemble_model.fit(train)
        pred_naiveensemble = naiveensemble_model.predict(term).pd_dataframe().reset_index()
        return pred_naiveensemble

    def arima(data, p, d, q, term):
        model = ARIMA(diff_log_earnings, order=(p, d, q))
        results = model.fit()
        forecast = results.get_forecast(steps=term)
        forecast_index = pd.date_range(earnings_series.index[-1], periods=term + 1, freq='M')[1:]
        forecast_values = np.exp(np.cumsum(forecast.predicted_mean))
        forecast_series = pd.Series(forecast_values.values, index=forecast_index)
        return forecast_series

    #def LR?
