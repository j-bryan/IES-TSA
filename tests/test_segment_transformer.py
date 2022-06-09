import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tsmodel.segmentation import SegmentTransformer, Segmenter, Concatenator
from tsmodel.models.arma import ARIMA
from tsmodel.transformers import FourierDetrend, KDENormalizer


class TestSegmentTransformer:
    def test_fit_transform_shape_univar_row(self):
        st = SegmentTransformer([('passthrough', None)])
        X = [np.arange(10) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert np.allclose(X, Xt)

    def test_fit_transform_shape_univar_col(self):
        st = SegmentTransformer([('passthrough', None)])
        X = [np.arange(10).reshape(-1, 1) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert np.allclose(X, Xt)

    def test_fit_transform_shape_multivar(self):
        st = SegmentTransformer([('passthrough', None)])
        X = [np.arange(20).reshape(10, 2) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert np.allclose(X, Xt)

    def test_inverse_transform(self):
        st = SegmentTransformer([('passthrough', None)])
        X = [np.arange(20).reshape(10, 2) for _ in range(5)]
        Xt = st.fit_transform(X)
        Xhat = st.inverse_transform(X)
        assert np.allclose(X, Xhat)

    def test_fdarima_univar_row(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=50) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert len(X) == len(Xt)
        for i in range(len(X)):
            assert X[i].shape == Xt[i].shape
    
    def test_fdarima_univar_row_inverse(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=50) for _ in range(5)]
        Xt = st.fit_transform(X)
        Xhat = st.inverse_transform(Xt)
        
        assert len(X) == len(Xhat)
        for i in range(len(X)):
            assert X[i].shape == Xhat[i].shape

    def test_fdarima_univar_col(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 1)) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert len(X) == len(Xt)
        for i in range(len(X)):
            assert X[i].shape == Xt[i].shape
    
    def test_fdarima_univar_col_inverse(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 1)) for _ in range(5)]
        Xt = st.fit_transform(X)
        Xhat = st.inverse_transform(Xt)
        
        assert len(X) == len(Xhat)
        for i in range(len(X)):
            assert X[i].shape == Xhat[i].shape

    def test_fdarima_multivar(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 2)) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert len(X) == len(Xt)
        for i in range(len(X)):
            assert X[i].shape == Xt[i].shape
    
    def test_fdarima_multivar_inverse(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        # st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
        #                          ('segment_arma', ARIMA((1, 0, 0)))])
        st = SegmentTransformer([('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 2)) for _ in range(5)]
        Xt = st.fit_transform(X)
        Xhat = st.inverse_transform(Xt)
        
        assert len(X) == len(Xhat)
        for i in range(len(X)):
            assert X[i].shape == Xhat[i].shape

    def test_fdarima_multivar_twotransforms(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
                                 ('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 2)) for _ in range(5)]
        Xt = st.fit_transform(X)
        assert len(X) == len(Xt)
        for i in range(len(X)):
            assert X[i].shape == Xt[i].shape
    
    def test_fdarima_multivar_inverse_twotransforms(self):
        """ Not ideal, since we'd like to isolate errors here from errors in the transformations used here, but
        we need to see if it handles multiple transform steps well. If tests for FourierDetrend or ARIMA are
        failing, be suspicious of failing here! """
        st = SegmentTransformer([('segment_fourier', FourierDetrend([50])),
                                 ('segment_arma', ARIMA((1, 0, 0)))])
        X = [np.random.normal(size=(50, 2)) for _ in range(5)]
        Xt = st.fit_transform(X)
        Xhat = st.inverse_transform(Xt)
        
        assert len(X) == len(Xhat)
        for i in range(len(X)):
            assert X[i].shape == Xhat[i].shape
    
    def test_ercot(self):
        # df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)
        # X = df[['TOTALLOAD', 'WIND']].to_numpy()
        X = np.random.normal(size=(8760, 2))
        t = np.arange(len(X))

        model = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                            ('kde_normalize', KDENormalizer()),
                            ('segment', Segmenter(pivot_length=72)),
                            # ('segment_transformer', SegmentTransformer([('passthrough', None)])),
                            # ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([24, 12])),
                            #                                             ('segment_arma', ARIMA((2, 0, 1)))])),
                            # ('concat', Concatenator()),
                            ('passthrough', None)])
        
        Xt = model.fit_transform(X)
        # assert len(X) == len(Xt)
        assert len(X) == np.sum(len(seg) for seg in Xt)
