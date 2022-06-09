from calendar import month
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.stats.diagnostic import het_arch

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from tsmodel.rom import ROM
from tsmodel.transformers import FourierDetrend, KDENormalizer, CDFPreserve, ColumnTransformer
from tsmodel.segmentation import Segmenter, Concatenator, SegmentTransformer
from tsmodel.models.arma import ARIMA
from tsmodel.models.garch import ARIMA_GARCH
from tsmodel.models.regime_switching import MarkovAR


def demo_univar_confint():
    """ Demonstrates fitting a model Pipeline with segmentation (with unequal segment lengths). """
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24

    X = df['TOTALLOAD'].to_numpy().reshape(-1, 1)
    t = np.arange(len(X))
    # model = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
    #                   ('kde_normalize', KDENormalizer()),
    #                   ('segment', Segmenter(pivot_length=72)),
    #                   ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([24, 12])),
    #                                                               ('segment_arma', ARIMA((2, 0, 1)))])),
    #                   ('concat', Concatenator()),
    #                   ('passthrough', None)])

    pipe1 = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                      ('kde_normalize', KDENormalizer())])
    pipe2 = Pipeline([('segment', Segmenter(seg_lens=months)),
                      ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([24, 12])),
                                                                  ('segment_arma', ARIMA((2, 0, 1)))])),
                      ('concat', Concatenator()),
                      ('passthrough', None)])

    print('Fitting model')
    # Xt = model.fit_transform(X)
    Xt1 = pipe1.fit_transform(X)
    Xt2 = pipe2.fit_transform(Xt1)

    print('Generating new sample paths')
    N_paths = 50
    synths = np.zeros((N_paths, *(X.shape)))
    for i in range(N_paths):
        print(f'\t{i+1}/{N_paths}')
        synths[i] = pipe2.inverse_transform(Xt2)
    synth_mean = np.mean(synths, axis=0)
    synth_std = np.std(synths, ddof=1, axis=0)

    # Xs_mean = pipe1.inverse_transform(synth_mean)
    # Xs_upper = pipe1.inverse_transform(synth_mean + 1.96 * synth_std).ravel()
    # Xs_lower = pipe1.inverse_transform(synth_std - 1.96 * synth_std).ravel()

    print('How often does the historical path fall within 2 std dev of the synthetic history mean?')
    # print(len(np.argwhere((X.ravel() <= synth_mean + 1.96 * synth_std) & (X.ravel() >= synth_mean - 1.96 * synth_std))) / len(X) * 100, '%')
    print(len(np.argwhere((Xt1 <= synth_mean + 1.96 * synth_std) & (Xt1 >= synth_mean - 1.96 * synth_std))) / len(Xt1) * 100, '%')

    plt.figure(1)
    plt.title('ERCOT Total Demand, 2021')
    # plt.plot(synth_mean, color='b', label='synth mean (95%CI)')
    # plt.fill_between(t, synth_mean - 1.96 * synth_std, synth_mean + 1.96 * synth_std, color='b', alpha=0.2)
    # plt.plot(Xs_mean, color='b', label='synth mean (95%CI)')
    # plt.fill_between(t, Xs_lower, Xs_upper, color='b', alpha=0.2)
    # plt.plot(X, color='orange', label='real')
    plt.plot(synth_mean, color='b', label='synth mean (95%CI)')
    plt.fill_between(t, synth_mean.ravel() - 1.96 * synth_std.ravel(), synth_mean.ravel() + 1.96 * synth_std.ravel(), color='b', alpha=0.2)
    plt.plot(Xt1, color='orange', label='real')
    plt.legend()

    plt.figure(2)
    plt.plot(Xt2)
    plt.xlabel('Time (h)')
    plt.ylabel('Normalized Residual')
    plt.title('ERCOT Total Demand, 2021')
    # plt.savefig('./plots/ERCOT/normresid_load.png', dpi=150)

    plt.show()


def demo_univar():
    """ Demonstrates fitting a model Pipeline with segmentation (with unequal segment lengths). """
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df['TOTALLOAD'].to_numpy().reshape(-1, 1)
    t = np.arange(len(X))
    model = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                      ('kde_normalize', KDENormalizer()),
                      ('segment', Segmenter(pivot_length=2*168)),
                      ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([24, 12])),
                                                                  ('segment_arma', ARIMA((2, 0, 1)))])),
                      ('concat', Concatenator()),
                      ('passthrough', None)])

    Xt = model.fit_transform(X)
    X_synth = model.inverse_transform(Xt)

    # plt.figure(1)
    # # plt.title('ERCOT Wind, 2021')
    # plt.plot(X_synth, label='synthetic')
    # plt.plot(X, color='orange', label='historical')
    # plt.legend()

    plt.figure(2)
    plt.plot(Xt)
    plt.xlabel('Time (h)')
    plt.ylabel('Normalized Residual')
    # plt.title('ERCOT Wind, 2021')
    plt.savefig('./plots/ERCOT/normresid_load_2weeksegments.png', dpi=150)

    plt.show()


def demo_univar_garch():
    """ Demonstrates fitting a model Pipeline with segmentation (with unequal segment lengths). """
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df['TOTALLOAD'].to_numpy()
    t = np.arange(len(X))
    # model = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
    #                   ('kde_normalize', KDENormalizer()),
    #                   ('segment', Segmenter(pivot_length=72)),
    #                   ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([24, 12]))])),
    #                   ('concat', Concatenator()),
    #                   ('garch', ARIMA_GARCH(arima_order=(3, 0, 2))),
    #                   ('passthrough', None)])
    preproc = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                        ('kde_normalize', KDENormalizer()),
                        ('segment', Segmenter(pivot_length=24)),
                        ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([12, 6, 3]))])),
                        ('concat', Concatenator()),
                        ('passthrough', None)])
    model = Pipeline([('garch', ARIMA_GARCH(arima_order=(1, 0, 0))),
                      ('passthrough', None)])
    model2 = Pipeline([('arma', ARIMA(order=(1, 0, 0))),
                       ('passthrough', None)])
    model3 = Pipeline([('arma', MarkovAR(k_regimes=4, order=1)),
                       ('passthrough', None)])

    X_resid = preproc.fit_transform(X)

    Xt = model.fit_transform(X_resid)
    Xt2 = model2.fit_transform(X_resid)
    Xt3 = model3.fit_transform(X_resid)

    X_synth_resid = model.inverse_transform(Xt)
    X_synth_resid_arma = model2.inverse_transform(Xt2)
    X_synth_resid_mar = model3.inverse_transform(Xt3)
    
    X_synth_garch = preproc.inverse_transform(X_synth_resid)
    X_synth_arma = preproc.inverse_transform(X_synth_resid_arma)
    X_synth_markov = preproc.inverse_transform(X_synth_resid_mar)

    plt.figure(1)
    plt.title('ERCOT Load, 2021')
    plt.plot(X_synth_garch, label='arma-garch')
    plt.plot(X_synth_arma, label='arma')
    plt.plot(X_synth_markov, label='markov switching')
    plt.plot(X, color='orange', label='historical')
    plt.legend()

    plt.figure(2)
    plt.plot(X_resid, label='Historical')
    plt.plot(X_synth_resid, label='ARMA-GARCH')
    plt.plot(X_synth_resid_arma, label='ARMA')
    plt.xlabel('Time (h)')
    plt.ylabel('Normalized Residual')
    plt.legend()
    plt.title('ERCOT Load, 2021')
    # plt.savefig('./plots/ERCOT/normresid_load.png', dpi=150)

    plt.show()


def demo_multivar():
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df[['TOTALLOAD', 'WIND']].to_numpy()
    t = np.arange(len(X))

    month_segments = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
    quarterly_segments = np.array([31 + 28 + 31, 30 + 31 + 30, 31 + 31 + 30, 31 + 30 + 31]) * 24
    months = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) * 24)
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    kde_column_transformer = ColumnTransformer([('col1', KDENormalizer(), [0]),
                                                ('col2', KDENormalizer(), [1])])
    segment_transformer = SegmentTransformer([('segment_fourier', FourierDetrend([12, 6, 3])),
                                              ('segment_arma', ARIMA((1, 0, 0)))])
    model = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                      ('kde_normalize', kde_column_transformer),
                      ('segment', Segmenter(pivot_length=24)),
                      ('segment_transformer', segment_transformer),
                      ('concat', Concatenator()),
                      ('passthrough', None)])

    Xt = model.fit_transform(X)
    X_synth = model.inverse_transform(Xt)

    plt.figure(1)
    plt.plot(t, X_synth[:, 0], label='synth')
    plt.plot(t, X[:, 0], label='real')
    plt.xticks(ticks=months, labels=labels)
    plt.ylabel('Total Load, MWh')
    plt.title('ERCOT Total Load, 2021\nSegments: 24 hours')
    plt.legend()
    plt.savefig('./plots/ercot_load_synth_daily.png', dpi=150)

    # plt.figure(2)
    # plt.plot(t, X_synth[:, 1], label='synth')
    # plt.plot(t, X[:, 1], label='real')
    # plt.ylabel(r'Wind Generation, $\%$ of Capacity')
    # plt.xticks(ticks=months, labels=labels)
    # plt.legend()

    plt.show()


def demo_mar():
    """ Demonstrates fitting a model Pipeline with segmentation (with unequal segment lengths). """
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv', index_col=0)

    X = df['TOTALLOAD'].to_numpy()
    t = np.arange(len(X))

    preproc = Pipeline([('full_series_fourier', FourierDetrend([8760, 4380, 2920, 2190, 438, 168, 24, 12, 6, 3])),
                        ('kde_normalize', KDENormalizer()),
                        ('segment', Segmenter(pivot_length=24)),
                        ('segment_transformer', SegmentTransformer([('segment_fourier', FourierDetrend([12, 6, 3]))])),
                        ('concat', Concatenator()),
                        ('passthrough', None)])

    model3 = Pipeline([('arma', MarkovAR(k_regimes=4, order=1, switching_variance=True)),
                       ('passthrough', None)])

    X_resid = preproc.fit_transform(X)
    Xt3 = model3.fit_transform(X_resid)
    X_synth_resid_mar = model3.inverse_transform(Xt3)
    X_synth_markov = preproc.inverse_transform(X_synth_resid_mar)

    plt.figure(1)
    plt.title('ERCOT Load, 2021')
    plt.plot(X_synth_markov, label='markov switching')
    plt.plot(X, color='orange', label='historical')
    plt.legend()

    plt.figure(2)
    plt.plot(X_resid, label='Historical')
    plt.plot(X_synth_resid_mar, label='Markov')
    plt.xlabel('Time (h)')
    plt.ylabel('Normalized Residual')
    plt.legend()
    plt.title('ERCOT Load, 2021')
    # plt.savefig('./plots/ERCOT/normresid_load.png', dpi=150)

    plt.show()


if __name__ == '__main__':
    # demo_univar()
    # demo_univar_garch()
    demo_mar()
    # demo_univar_wind()
    # demo_multivar()
