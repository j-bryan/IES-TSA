import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram


def main():
    df = pd.read_csv('./data/ERCOT_2021/Data_0.csv')
    f, Pxx = periodogram(df['WIND'])
    plt.loglog(1 / f[1:], Pxx[1:])
    plt.show()


if __name__ == '__main__':
    main()
