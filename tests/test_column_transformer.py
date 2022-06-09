import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsmodel.transformers import FourierDetrend
from tsmodel.transformers import ColumnTransformer


class TestColumnTransformer:
    def test_roundtrip(self):
        X = np.random.normal(size=(100, 2))
        # col_trans = ColumnTransformer([('col1', MinMaxScaler(), [0]),
        #                                ('col2', MinMaxScaler(), [1])])
        col_trans = ColumnTransformer([('col1', FourierDetrend([1]), [0]),
                                       ('col2', FourierDetrend([1]), [1])])
        Xt = col_trans.fit_transform(X)
        # Xt = col_trans.transform(X)
        Xhat = col_trans.inverse_transform(Xt)
        assert np.allclose(X, Xhat)
