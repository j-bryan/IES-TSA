import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch.univariate import arch_model
from sklearn.base import BaseEstimator, TransformerMixin


class ARIMA_GARCH(BaseEstimator, TransformerMixin):
    def __init__(self, arima_order=(1, 0, 0)) -> None:
        self.arima_order = arima_order        
        self.arima = None
        self.garch = None
        self.garch_params = []

    def fit(self, X, y=None, **fit_params):
        self.arima = ARIMA(X, order=self.arima_order).fit(**fit_params)
        p_ = self.arima_order[0]
        o_ = self.arima_order[1]
        q_ = self.arima_order[2]
        self.garch = arch_model(self.arima.resid, p=p_, o=o_, q=q_, dist='StudentsT')
        garch_fit_res = self.garch.fit()
        self.garch_params = garch_fit_res.params
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, Xt):
        # generate synthetic history here
        eps = self.garch.simulate(self.garch_params, len(Xt))['data'].to_numpy()
        mu = self.arima.simulate(nsimulations=len(Xt))
        return mu + eps
