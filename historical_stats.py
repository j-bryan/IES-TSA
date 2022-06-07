"""
TODO: I want to run stuff like Breusch-Pagan here.

Breusch-Pagan shouldn't be run in the other analysis, as the test should be applied specifically to the historical data.
Also, I'll have to use my own detrending and normalization code for this, since we need to apply the test to the normalized
residual. RAVEN can't provide that.

TODO: Add my code from Google Drive that has this already
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.stats.diagnostic import het_arch

from sklearn.pipeline import Pipeline

from tsmodel.rom import ROM
from tsmodel.transformers import FourierDetrend, KDENormalizer, CDFPreserve
from tsmodel.segmentation import Segmenter, Concatenator
from tsmodel.models.arma import ARIMA


def demo_univar():
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df['TOTALLOAD'].to_numpy().reshape(-1, 1)
    t = np.arange(len(X))
    preprocessor = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                             ('kde_normalize', KDENormalizer()),
                             ('segment', Segmenter(pivot_length=24)),
                             ('segment_fourier', FourierDetrend([12, 6, 3])),
                             ('concat', Concatenator()),
                             ('passthrough', 'passthrough')])

    # rom = ROM(preprocessor, ARIMA(order=(3, 0, 2)), postprocessor=CDFPreserve())
    rom = ROM(preprocessor, ARIMA(order=(3, 0, 2)))
    rom.fit(X)

    X_synth = rom.simulate()
    plt.figure(1)
    plt.plot(X_synth, label='synth')
    plt.plot(X, label='real')
    plt.legend()

    plt.show()


def demo_multivar():
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df[['TOTALLOAD', 'WIND']].to_numpy()
    t = np.arange(len(X))
    preprocessor = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                            #  ('kde_normalize', KDENormalizer()),
                             ('segment', Segmenter(pivot_length=24)),
                             ('segment_fourier', FourierDetrend([12, 6, 3])),
                             ('concat', Concatenator()),
                             ('passthrough', 'passthrough')])

    # rom = ROM(preprocessor, ARIMA(order=(3, 0, 2)), postprocessor=CDFPreserve())
    rom = ROM(preprocessor, ARIMA(order=(3, 0, 2)))
    rom.fit(X)

    X_synth = rom.simulate()
    plt.figure(1)
    plt.plot(X_synth[:, 0], label='synth')
    plt.plot(X[:, 0], label='real')
    plt.legend()

    plt.figure(2)
    plt.plot(X_synth[:, 1], label='synth')
    plt.plot(X[:, 1], label='real')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    demo_univar()
