import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


class MarkovAR(TransformerMixin, BaseEstimator):
    def __init__(self, k_regimes, order, **kwargs) -> None:
        super().__init__()
        self.k_regimes = k_regimes
        self.order = order
        self.model = None
        self._fit_result = None
        self._nobs = 0
        self._input_shape = None
        self._kwargs = kwargs
        
    def fit(self, X, y=None, **fit_params):
        self._input_shape = X.shape
        if X.ndim > 1 and X.shape[1] > 1:  # multivariate series but a univariate model!
            self.model = [MarkovAutoregression(X[:, i], k_regimes=self.k_regimes, order=self.order, **self._kwargs).fit(**fit_params) for i in range(X.shape[1])]
        else:
            self.model = MarkovAutoregression(X, k_regimes=self.k_regimes, order=self.order, **self._kwargs).fit(**fit_params)
            print(self.model.summary())
        self._nobs = len(X)
        return self

    def transform(self, X):
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
