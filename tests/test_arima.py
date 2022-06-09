import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from tsmodel.models.arma import ARIMA


class TestARIMA:
    def test_univar_output_shape_row(self):
        arma = ARIMA((1, 0, 0))
        
        X = np.random.normal(size=100)
        Xt = arma.fit_transform(X)
        X_synth = arma.inverse_transform(Xt)

        assert not np.allclose(X, X_synth)  # make sure the same vector didn't get returned
        assert X.shape == X_synth.shape
    
    def test_univar_output_shape_col(self):
        arma = ARIMA((1, 0, 0))
        
        X = np.random.normal(size=(100, 1))
        Xt = arma.fit_transform(X)
        X_synth = arma.inverse_transform(Xt)

        assert not np.allclose(X, X_synth)  # make sure the same vector didn't get returned
        assert X.shape == X_synth.shape
    
    def test_multivar_output_shape(self):
        arma = ARIMA((1, 0, 0))
        
        X = np.random.normal(size=(100, 3))
        Xt = arma.fit_transform(X)
        X_synth = arma.inverse_transform(Xt)

        assert not np.allclose(X, X_synth)  # make sure the same vector didn't get returned
        assert X.shape == X_synth.shape
    
    def test_transform(self):
        arma = ARIMA((1, 0, 0))
        
        X = np.random.normal(size=(100, 1))
        Xt = arma.fit_transform(X)
        
        assert np.allclose(X, Xt)  # make sure the same vector didn't get returned
