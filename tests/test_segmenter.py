import numpy as np
from tsmodel.segmentation import Segmenter, Concatenator

import pytest


class TestSegmenter:
    def test_univariate_row_fittransform(self):
        X = np.arange(10)
        seg = Segmenter(pivot_length=5)
        
        X_seg = seg.fit_transform(X)
        X_seg_true = X.reshape(2, 5)

        assert np.allclose(X_seg, X_seg_true)
    
    def test_univariate_col_fittransform(self):
        X = np.arange(10).reshape(-1, 1)
        seg = Segmenter(pivot_length=5)
        
        X_seg = seg.fit_transform(X)
        X_seg_true = X.reshape(2, 5, 1)

        assert np.allclose(X_seg, X_seg_true)
    
    def test_univariate_row_seglens(self):
        X = np.arange(10)
        seg = Segmenter(seg_lens=[4, 4, 2])
        
        X_seg = seg.fit_transform(X)
        X_seg_true = [X[:4], X[4:8], X[8:]]

        for i in range(len(X_seg)):  # np.allclose can't handle a list that can't be put into a numpy array
            assert np.allclose(X_seg[i], X_seg_true[i])
    
    def test_univariate_inverse_transform(self):
        X = np.arange(10)
        seg = Segmenter(pivot_length=5)
        
        X_seg = seg.fit_transform(X)
        Xhat = seg.inverse_transform(X_seg)
        print(X_seg)
        print(Xhat)
        print(X)

        assert np.allclose(X, Xhat)
    
    def test_multivariate_fittransform(self):
        X = np.arange(30).reshape((3, 10)).T
        seg = Segmenter(pivot_length=5)

        X_seg = seg.fit_transform(X)
        X_seg_true = X.reshape((2, 5, 3))

        assert np.allclose(X_seg, X_seg_true)
    
    def test_multivariate_inverse_transform(self):
        X = np.arange(30).reshape((3, 10)).T
        seg = Segmenter(pivot_length=5)

        X_seg = seg.fit_transform(X)
        Xhat = seg.inverse_transform(X_seg)

        assert np.allclose(X, Xhat)

    def test_multivariate_seglens(self):
        X = np.arange(30).reshape((3, 10)).T
        seg = Segmenter(seg_lens=[4, 4, 2])

        X_seg = seg.fit_transform(X)
        X_seg_true = [X_seg[:4], X_seg[4:8], X_seg[8:]][0]

        for i in range(len(X_seg)):  # np.allclose can't handle a list that can't be put into a numpy array
            assert np.allclose(X_seg[i], X_seg_true[i])
    
    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            seg = Segmenter()


class TestConcatenator:
    def test_fittransform_rows(self):
        c = Concatenator()
        X_seg = np.arange(10).reshape(2, 5)
        X_cat = c.fit_transform(X_seg)
        assert np.allclose(X_seg.ravel(), X_cat)
    
    def test_fittransform_cols(self):
        c = Concatenator()
        X_seg = np.arange(10).reshape(2, 5, 1)
        X_true = np.arange(10).reshape(-1, 1)
        X_cat = c.fit_transform(X_seg)
        assert np.allclose(X_true, X_cat)
