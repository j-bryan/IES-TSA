import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats.mstats import mquantiles
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.base import BaseEstimator, TransformerMixin

from pathos.multiprocessing import ProcessingPool as Pool


class ECDFNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ecdf = None

    def fit(self, X, y=None):
        self.ecdf = ECDF(X)
        return self

    def transform(self, X, y=None):
        if self.ecdf is None:
            self.fit(X.ravel(), y)

        x_ecdf = self.ecdf(X.ravel())

        # Can't have exactly 0 or 1 in the ECDF values or we'll get infinities in the transform
        x_ecdf[x_ecdf >= 1.0] = 1.0 - np.finfo(float).eps
        x_ecdf[x_ecdf <= 0.0] = np.finfo(float).eps

        # Transform to the standard normal distribution
        x_trans = norm.ppf(x_ecdf)

        return x_trans

    def inverse_transform(self, X, y=None):
        # TODO
        pass


class KDENormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.kde = None
        self.cdf = None
        self.icdf = None

    def fit(self, X, y=None):
        self.cdf = self._cdf_approx(X.ravel())
        self.icdf = self._icdf_approx()
        return self

    def transform(self, X, y=None, **kwargs):
        if self.cdf is None:
            self.fit(X.ravel(), y)
        X_norm = norm.ppf(self.cdf(X.ravel()))
        return X_norm.reshape(-1, 1)

    def inverse_transform(self, X, y=None, **kwargs):
        if self.icdf is None:
            self.fit(X.ravel(), y)
        X_origdist = self.icdf(norm.cdf(X.ravel()))
        return X_origdist.reshape(-1, 1)

    def _cdf_approx(self, X):
        """ Approximates the CDF of a dataset. This is done according to the following steps:
                1. Approximate the PDF with kernel density estimation (KDE)
                2. Determine a suitable range of points from which to build a CDF.
                3. Calculate the CDF at each of these values.
                4. Interpolate these values to create a callable CDF function.
        """
        # Run KDE on the given data to approximate the PDF of the data. Using the statsmodels module for this.
        self.kde = sm.nonparametric.KDEUnivariate(X)
        self.kde.fit()

        # Time to work on creating a CDF function!
        # By default, we'll use 1.5x the mean-extreme range. This usually corresponds to about 6 standard deviations from the mean.
        # It's important to go a bit beyond what we think we'll need, since we want to interpolate between the CDF values we calculate.
        mean = np.mean(X)
        lb = mean - 1.2 * abs(min(X) - mean)
        ub = mean + 1.2 * abs(max(X) - mean)

        pts = np.linspace(lb, ub, 100)

        # Calculate CDF values for a number of points between the minimum and maximum values. We'll use multiprocessing for
        # this so it doesn't take so long.
        # TODO: also add in serial processing option for this!
        def f(bounds):
            a, b = bounds
            y = quad(self.kde.evaluate, a, b)[0]
            return y

        args = np.vstack(([-np.inf, *pts[:-1]], pts)).T

        with Pool() as p:
            probs = np.asarray(p.map(f, args))

        cumprobs = np.cumsum(probs)

        # Interpolate these CDF values so we have a callable function. We'll do this with monotonic cubic spline interpolation.
        cdf = PchipInterpolator(pts, cumprobs)

        return cdf

    def _icdf_approx(self):
        # quantile values already calculated in self.kde.icdf
        x = np.linspace(0, 1, len(self.kde.density))
        icdf = PchipInterpolator(x, self.kde.icdf)
        return icdf
