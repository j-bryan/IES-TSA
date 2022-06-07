import numpy as np
from statsmodels.tsa.arima.model import ARIMA as arima
from tsmodel.models.base import TimeSeriesBase


class ARIMA(TimeSeriesBase):
    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0)) -> None:
        self._order = order
        self._seasonal_order = seasonal_order
        self.model = None
        self._fit_result = None
        self._nobs = 0
        self._input_shape = None
        
    def fit(self, X, y=None, **fit_params):
        self._input_shape = X.shape
        if X.ndim > 1 and X.shape[1] > 1:  # multivariate series but a univariate model!
            self.model = [arima(X[:, i], order=self._order, seasonal_order=self._seasonal_order).fit(**fit_params) for i in range(X.shape[1])]
        else:
            self.model = arima(X, order=self._order, seasonal_order=self._seasonal_order).fit(**fit_params)
        self._nobs = len(X)
        return self

    def simulate(self, **sim_params):
        assert self.model is not None
        if isinstance(self.model, list):
            return np.stack([model.simulate(nsimulations=self._nobs, **sim_params) for model in self.model])
        else:
            return self.model.simulate(nsimulations=self._nobs, **sim_params)

    def summary(self):
        return self._fit_result.summary()
    
    def is_univariate(self):
        return True
