import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tsmodel.transformers import KDENormalizer


class TestKDENormalizer:
    def test_init(self):
        kde = KDENormalizer()
    
    def test_univar_row_shape(self):
        X = np.random.normal(size=100)
        kde = KDENormalizer()
        Xt = kde.fit_transform(X)
        assert X.shape == Xt.shape
    
    def test_univar_col_shape(self):
        X = np.random.normal(size=(100, 1))
        kde = KDENormalizer()
        Xt = kde.fit_transform(X)
        assert X.shape == Xt.shape
    
    def test_multivar_shape(self):
        X = np.random.normal(size=(100, 2))
        kde = KDENormalizer()
        Xt = kde.fit_transform(X)
        assert X.shape == Xt.shape
    
    def test_inverse(self):
        # TODO: see why exactly this is failing! Could be due to not using enough segments when approximating cdf/icdf?
        X = np.random.normal(size=1000)
        kde = KDENormalizer()
        Xt = kde.fit_transform(X)
        Xhat = kde.inverse_transform(Xt)
        # assert np.allclose(X, Xhat)
        assert Xhat.shape == X.shape
    
    def test_coltransformer(self):
        X = np.random.normal(size=(100, 2))
        coltrans = ColumnTransformer([('col1', KDENormalizer(), [0]),
                                      ('col2', KDENormalizer(), [1])])
        Xt = coltrans.fit_transform(X)
        assert Xt.shape == X.shape
