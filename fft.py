import numpy as np
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import pandas as pd


def ercot():
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv')

    names = {
        'TOTALLOAD': 'Total Load',
        'WIND': 'Wind Power',
        'SOLAR': 'Solar Power'
    }

    for var, label in names.items():
        f, Pxx = periodogram(df[var].to_numpy())
        plt.figure()
        plt.loglog(1 / f[1:], Pxx[1:])
        plt.xlabel(label)
    plt.show()


if __name__ == '__main__':
    ercot()
