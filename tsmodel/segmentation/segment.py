import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Segmenter(BaseEstimator, TransformerMixin):
    def __init__(self, pivot_length):
        self.pivot_length = pivot_length
        self.nobs = 0
        self.n_segments = 0  # number of segments
    
    def fit(self, X, y=None, **fit_params):
        self.nobs = len(X)
        self.n_segments = self.nobs // self.pivot_length
        if self.n_segments * self.pivot_length != len(X):
            warnings.warn(f'Segmentation pivot length ({self.pivot_length}) does not evenly divide '\
                            'the array of shape {X.shape}. Wrapping array to match size.')
            self.n_segments += 1
        return self

    def transform(self, X):
        return X.take(np.arange(self.n_segments * self.pivot_length), mode='wrap').reshape((n_records, self.pivot_length))

    def inverse_transform(self, Xt):
        return Xt.ravel()[:self.nobs]
