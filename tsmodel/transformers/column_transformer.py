import numpy as np
# from sklearn.compose import ColumnTransformer as ColTrans
from sklearn.base import BaseEstimator, TransformerMixin


# class ColumnTransformer(ColTrans):
#     def __init__(self, transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False):
#         super().__init__(transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
#                          transformer_weights=transformer_weights, verbose=verbose)
#         for t in transformers:
#             assert t[2] != 'drop'
    
#     def inverse_transform(self, Xt):
#         X = np.zeros(Xt.shape)
#         for transformer in self.transformers:
#             name, estimator, cols = transformer
#             X[:, cols] = estimator.inverse_transform(Xt[:, cols])
#         return X


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers) -> None:
        super().__init__()
        self.transformers = transformers
    
    def fit(self, X, y=None, **fit_params):
        for i in range(len(self.transformers)):
            self.transformers[i][1].fit(X, y, **fit_params)
        return self
    
    def transform(self, X):
        Xt = np.zeros(X.shape)
        for i in range(len(self.transformers)):
            name, estimator, cols = self.transformers[i]
            Xt[:, cols] = estimator.transform(X[:, cols])
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        Xt = np.zeros(X.shape)
        for i in range(len(self.transformers)):
            # name, estimator, cols = self.transformers[i]
            Xt[:, self.transformers[i][2]] = self.transformers[i][1].fit_transform(X[:, self.transformers[i][2]])
        return Xt
    
    def inverse_transform(self, Xt):
        X = np.zeros(Xt.shape)
        for i in range(len(self.transformers)):
            name, estimator, cols = self.transformers[i]
            X[:, cols] = estimator.inverse_transform(Xt[:, cols])
        return X
