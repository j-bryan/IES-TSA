from statsmodels.tsa.arima.model import ARIMA as arima
from models.base import TimeSeriesBase


class ARIMA(TimeSeriesBase):
    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0)) -> None:
        self._order = order
        self._seasonal_order = seasonal_order
        self.model = None
        self._fit_result = None
        self._nobs = 0
    
    def fit(self, X, y=None, **fit_params):
        self.model = arima(X, order=self._order, seasonal_order=self._seasonal_order).fit(**fit_params)
        self._nobs = len(X)
        return self

    def simulate(self, **sim_params):
        assert self.model is not None
        return self.model.simulate(nsimulations=self._nobs, **sim_params)

    def summary(self):
        return self._fit_result.summary()
