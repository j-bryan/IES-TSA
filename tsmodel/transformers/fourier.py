import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import periodogram


class FourierDetrendBase(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._F = None

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        return X - self._F

    def inverse_transform(self, Xt, y=None):
        return Xt + self._F

    @staticmethod
    def _fourier_lstsq(x, c, k):
        """ Fits a given set of Fourier modes given by c and k to the signal x """
        t = np.arange(x.size)

        P = np.zeros((len(c), k))  # periods are all (c_i / f_j)
        for i, ci in enumerate(c):
            for j in range(k):
                P[i, j] = ci / (j + 1)
        P = P.ravel()

        G = np.hstack(
            (np.ones((len(t), 1)), np.sin(2 * np.pi / P * t.reshape(-1, 1)), np.cos(2 * np.pi / P * t.reshape(-1, 1))))
        m = np.linalg.pinv(G) @ x

        xhat = G @ m

        return xhat


class FFTDetrend(FourierDetrendBase):
    def __init__(self, nfreq):
        super().__init__()
        self._nfreq = nfreq

    def fit(self, X, y=None):
        f, Pxx = periodogram(X.ravel())
        inds = np.argpartition(Pxx, -self._nfreq)[-self._nfreq:]
        self._F = self._fourier_lstsq(X, 1/f[inds], 1).reshape(-1, 1)
        return self


class FourierDetrend(FourierDetrendBase):
    def __init__(self, c, k=1):
        super().__init__()
        self._c = c
        self._k = k

    def fit(self, X, y=None):
        self._F = self._fourier_lstsq(X, self._c, self._k)
        return self
