from lib2to3.pytree import Base
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as arima
from sklearn.base import TransformerMixin, BaseEstimator
from tsmodel.base import Clusterable
from tsmodel.models.base import TimeSeriesBase


class ARIMA(TransformerMixin, BaseEstimator, TimeSeriesBase, Clusterable):
    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 0), cluster=False) -> None:
        super().__init__()
        self._order = order
        self._seasonal_order = seasonal_order
        self.model = None
        self._fit_result = None
        self._nobs = 0
        self._input_shape = None
        self._cluster = cluster
        
    def fit(self, X, y=None, **fit_params):
        self._input_shape = X.shape
        if X.ndim > 1 and X.shape[1] > 1:  # multivariate series but a univariate model!
            self.model = [arima(X[:, i], order=self._order, seasonal_order=self._seasonal_order).fit(**fit_params) for i in range(X.shape[1])]
            for m in self.model:
                self._clusterable_features.extend(m.params)
        else:
            self.model = arima(X, order=self._order, seasonal_order=self._seasonal_order).fit(**fit_params)
            self._clusterable_features = self.model.params
        self._nobs = len(X)
        return self

    def transform(self, X):
        if self._cluster:
            return self._clusterable_features
        else:
            return X
    
    def inverse_transform(self, Xt):
        X = np.zeros(Xt.shape)
        if isinstance(self.model, list):
            for i, m in enumerate(self.model):
                X[:, i] = m.simulate(nsimulations=len(Xt[:, i]))
        else:
            X = self.model.simulate(nsimulations=len(Xt)).reshape(Xt.shape)
        return X

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
