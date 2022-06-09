import numpy as np
from scipy.signal import periodogram
from sklearn.base import BaseEstimator, TransformerMixin
from tsmodel.base import Clusterable


class FourierDetrendBase(BaseEstimator, TransformerMixin, Clusterable):
    def __init__(self):
        super().__init__()
        self._m = None  # model parameters (i.e. amplitudes, intercept)
        self._P = None  # Fourier periods

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    def transform(self, X, y=None):
        t = np.arange(len(X)).reshape(-1, 1)
        return X - self.F(t)

    def inverse_transform(self, Xt, y=None):
        t = np.arange(len(Xt)).reshape(-1, 1)
        return Xt + self.F(t)

    # def _fourier_lstsq(self, X, c):
    #     """ Fits a given set of Fourier modes given by c and k to the signal x """
    #     t = np.arange(len(X)).reshape(-1, 1)
    #     self._P = np.asarray(c, dtype=float).ravel()
        
    #     # G = np.hstack((np.ones((len(t), 1)), np.sin(2 * np.pi * t @ (P ** -1)), np.cos(2 * np.pi * t @ (P ** -1))))
    #     G = np.hstack((np.ones((len(t), 1)), np.sin(2 * np.pi / P * t), np.cos(2 * np.pi / P * t)))
    #     self._m = np.linalg.pinv(G) @ X

    #     F = G @ self._m  # reconstructs Fourier signal with estimated amplitudes
    #     self._clusterable_features.extend(m.ravel())

    #     return F
    
    def fit_fourier_params(self, X, c, **fit_params):
        t = np.arange(len(X)).reshape(-1, 1)
        self._P = np.asarray(c, dtype=float).ravel()
        G = np.hstack((np.ones((len(t), 1)), np.sin(2 * np.pi / self._P * t), np.cos(2 * np.pi / self._P * t)))
        self._m = np.linalg.pinv(G) @ X
    
    def F(self, t):
        G = np.hstack((np.ones((len(t), 1)), np.sin(2 * np.pi / self._P * t), np.cos(2 * np.pi / self._P * t)))
        return G @ self._m


class FFTDetrend(FourierDetrendBase):
    def __init__(self, nfreq):
        super().__init__()
        self._nfreq = nfreq

    def fit(self, X, y=None, **fit_params):
        f, Pxx = periodogram(X.ravel())  # TODO: won't work well for all input shapes
        inds = np.argpartition(Pxx, -self._nfreq)[-self._nfreq:]
        P = 1 / f[inds]
        self.fit_fourier_params(X, P)
        return self


class FourierDetrend(FourierDetrendBase):
    def __init__(self, c):
        super().__init__()
        self._P = np.asarray(c).ravel()

    def fit(self, X, y=None, **fit_params):
        self.fit_fourier_params(X, self._P)
        return self


class SegmentFourierDetrend(FourierDetrendBase):
    def __init__(self, c):
        super().__init__()
        self._c = c
        self._clusterable_features = []
    
    def _fourier_segmented(self, x, c):
        Fs = []
        ms = []
        for i, x_seg in enumerate(x):
            F, m = self._fourier_lstsq(x_seg, c)
            Fs.append(F)
            ms.append(m)
        return Fs
    
    def fit(self, X, y=None, **fit_params):
        self._F, self._m = self._fourier_segmented(X)
        self._clusterable_features.extend(list(self._F))
        self._clusterable_features.extend(list(self._m))
        return self

    def transform(self, X):
        return [Xi - Fi for (Xi, Fi) in zip(X, self._F)]

    def inverse_transform(self, Xt):
        return [Xt[i] + self._F[i] for i in range(len(Xt))]

    def get_clusterable_features(self):
        return self._clusterable_features
