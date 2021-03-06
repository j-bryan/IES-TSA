import numpy as np
from tsmodel.transformers.fourier import FourierDetrend


class TestFourierDetrend:
    def test_transform_sine(self):
        """ FourierDetrend should perfectly remove a perfect sinusoidal signal. Tested here
        with a single sine wave.
        """
        F = FourierDetrend(c=[1])
        t = np.arange(10)
        x = np.sin(2 * np.pi * t)
        xhat = F.fit_transform(x)
        print(xhat.shape)
        print(xhat)
        assert np.allclose(xhat, np.zeros(xhat.shape))
    
    def test_fourier_multi(self):
        """ A purely sinusoidal signal but with multiple frequencies and random amplitudes """
        c = np.array([1., 2, 3, 4, 5, 6, 7])
        F = FourierDetrend(c=c)

        A = np.random.normal(size=len(c))
        B = np.random.normal(size=len(c))
        
        t = np.arange(1000)
        x = np.zeros(t.shape)
        for i, ci in enumerate(c):
            x += A[i] * np.sin(2 * np.pi / ci * t) + B[i] * np.cos(2 * np.pi / ci * t)
        
        xhat = F.fit_transform(x)
        assert np.allclose(xhat, np.zeros(xhat.shape))
    
    def test_input_shape_rowvec(self):
        F = FourierDetrend(c=[1])
        x_row = np.linspace(0, 44/7, 100)
        xhat_row = F.fit_transform(x_row)
        assert xhat_row.shape == x_row.shape
    
    def test_input_shape_colvec(self):
        F = FourierDetrend(c=[1])
        x_col = np.linspace(0, 44/7, 100).reshape(-1, 1)
        xhat_col = F.fit_transform(x_col)
        assert xhat_col.shape == x_col.shape
    
    def test_input_shape_multivar(self):
        F = FourierDetrend(c=[1])
        x_multi = np.linspace(0, 44/7, 300).reshape(3, 100).T
        xhat_multi = F.fit_transform(x_multi)
        assert xhat_multi.shape == x_multi.shape
