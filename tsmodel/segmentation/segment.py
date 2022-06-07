import warnings
from collections.abc import Iterable
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Segmenter(BaseEstimator, TransformerMixin):
    def __init__(self, pivot_length=None, seg_lens=None):
        if pivot_length is None and seg_lens is None:
            raise ValueError('Either pivot_length or seg_lens must be specified!')
        
        if pivot_length:
            self._pivot_length = pivot_length
            self._seg_lens = []
        elif seg_lens:
            self._seg_lens = seg_lens
            self._pivot_length = None  # value won't actually be used if seg_lens is already defined
        else:
            raise ValueError('Either pivot_length or seg_lens must be specified!')
            
        self.nobs = 0
        self.n_segments = 0

    def fit(self, X, y=None, **fit_params):
        self.nobs = len(X)  # first array dimension must index over time steps for every input shape
        if self._pivot_length and not self._seg_lens:
            n_evensplits = self.nobs // self._pivot_length
            self._seg_lens = [self._pivot_length] * n_evensplits
            remainder = self.nobs % self._pivot_length
            if remainder != 0:
                self._seg_lens.append(remainder)

        self.n_segments = len(self._seg_lens)
        return self

    def transform(self, X):
        # Want to put the data X into an array of segments of the same dimension as X
        ##   X.shape = (n,)    -->  X_seg.shape = (n_segments, pivot_length)
        ##   X.shape = (n, 1)  -->  X_seg.shape = (n_segments, pivot_length, 1)
        ##   X.shape = (n, m)  -->  X_seg.shape = (n_segments, pivot_length, m)
        # return np.split(X, self._seg_lens)
        return np.array_split(X, np.cumsum(self._seg_lens[:-1]))

    def inverse_transform(self, Xt):
        return np.concatenate(Xt)

    def get_seglens(self):
        return self._seg_lens
