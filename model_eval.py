import os

import numpy as np
import scipy
import pandas as pd
import statsmodels as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

from statsmodels.graphics.gofplots import qqplot_2samples
from statsmodels.graphics.boxplots import violinplot
# import seaborn as sns
import matplotlib.pyplot as plt


class ModelAssessment:
    def __init__(self, df_h, df_s, model_params):
        self.df_h = df_h
        self.df_s = df_s
        self._zero_mask_vars = ['SOLAR']
        self.model_params = model_params
        # var_names = df_h.columns[1:]  # drop HOUR column
        var_names = list(df_h.columns)  # drop HOUR column
        var_names.remove('HOUR')
        self.names = var_names

        data = []
        for n in var_names:
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)

            ks_res = scipy.stats.ks_2samp(vals_h, vals_s)
            wd = scipy.stats.wasserstein_distance(vals_h, vals_s)
            
            # Calculate autocorrelation function for historical and synthetic data
            acf_h, acf_h_confint = acf(vals_h, nlags=24, alpha=0.05)
            acf_s, acf_s_confint = acf(vals_s, nlags=24, alpha=0.05)
            # If 0 is contained in the 95% CI, set that lag to 0
            acf_h[acf_h_confint.prod(axis=1) < 0] = 0
            acf_s[acf_s_confint.prod(axis=1) < 0] = 0
            # Calculate the squared error between the historical and synthetic data ACFs.
            acf_se = np.sum((acf_h - acf_s) ** 2)

            # Repeat for PACF
            pacf_h, pacf_h_confint = pacf(vals_h, nlags=24, alpha=0.05)
            pacf_s, pacf_s_confint = pacf(vals_s, nlags=24, alpha=0.05)
            pacf_h[pacf_h_confint.prod(axis=1) < 0] = 0
            pacf_s[pacf_s_confint.prod(axis=1) < 0] = 0
            pacf_se = np.sum((pacf_h - pacf_s) ** 2)

            data.append([n, ks_res.statistic, ks_res.pvalue, wd, acf_se, pacf_se])

        self.stats = pd.DataFrame(data, columns=['var', 'KS2samp_Fstat', 'KS2samp_pvalue', 'WassersteinDist', 'ACF_SE', 'PACF_SE'])

        wdir = model_params['WorkingDir'][1]
        figs_dir = os.path.join(model_params['WorkingDir'][1], 'figures')
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        self._qqplot(figs_dir)
        self._ecdf_plot(figs_dir)
        self._sample_plot(figs_dir)
        # self._violinplot(model_params['WorkingDir'][1])

    def _mask_zeros(self, v, n):
        v_tmp = np.asarray(v)
        if n in self._zero_mask_vars:
            return v_tmp[v_tmp != 0]
        else:
            return v_tmp

    def _get_param(self, k):
        if k in self.model_params:
            v = self.model_params.get(k)[1]
        elif k == 'pivotLength':
            v = self.model_params['subspace'][1]
        else:
            v = 0
        return v

    def _params_title(self):
        s = ''
        for k, v in self.model_params.items():
            if k == 'WorkingDir':
                continue
            if v[0] == '':
                attr_name = k
            else:
                attr_name = v[0]
            s += '({}: {}) '.format(attr_name, v[1])
        return s

    def _ecdf_plot(self, pth):
        fig, ax = plt.subplots(nrows=len(self.names))
        if len(self.names) == 1:
            ax = [ax]
        fig.suptitle(self._params_title())
        for i, n in enumerate(self.names):
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)
            ax[i].set_ylabel(n)
            ax[i].plot(np.sort(vals_h), np.linspace(0, 1, len(vals_h), endpoint=False), label='Hist')
            ax[i].plot(np.sort(vals_s), np.linspace(0, 1, len(vals_s), endpoint=False), label='Synth')
            ax[i].legend()
        plt.savefig(os.path.join(pth, 'ecdf.png'))
        plt.close(fig)

    def _sample_plot(self, pth):
        fig, ax = plt.subplots(nrows=len(self.names))
        df_s = self.df_s.query('RAVEN_sample_ID == 0')
        if len(self.names) == 1:
            ax = [ax]
        fig.suptitle(self._params_title())
        for i, n in enumerate(self.names):
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(df_s[n], n)
            ax[i].set_ylabel(n)
            ax[i].plot(vals_h, label='Hist')
            ax[i].plot(vals_s, label='Synth')
            ax[i].legend()
        plt.savefig(os.path.join(pth, 'samples.png'))
        plt.close(fig)

    def _qqplot(self, pth):
        fig, ax = plt.subplots(ncols=len(self.names))
        if len(self.names) == 1:
            ax = [ax]
        fig.suptitle(self._params_title())
        for i, n in enumerate(self.names):
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)
            ax[i].set_title(n)
            ax[i].set_aspect('equal')
            qqplot_2samples(vals_s, vals_h, xlabel='Synthetic', ylabel='Historical', line='45', ax=ax[i])
        plt.savefig(os.path.join(pth, 'qq.png'))
        plt.close(fig)

    def summarize(self):
        return ModelSummary(self)


class ModelSummary:
    """ Class to condense some of the data contained in a ModelAssessment object """
    def __init__(self, m):
        self.stats = m.stats
        self.model_params = m.model_params
        # self.qqplot = m.qqplot
        # self.violinplot = m.violinplot


class StatEnsemble:
    """ Class to aggregate summary statistics across multiple time series models """
    def __init__(self) -> None:
        self.summaries = []
        self.aggstats = None
    
    def append(self, m):
        self.summaries.append(m.summarize())
        self._resummarize(m)
    
    def _resummarize(self, m):
        # add whole ModelSummary.stats DataFrame to return df if all ModelSummary.model_params fall within the ranges given by params
        ## iterate over models
        #### iterate over parameters
        m_stats_df = m.stats
        for k in m.model_params.keys():
            try:
                val = float(m.model_params[k][1])
            except ValueError:  # can't convert string to float
                continue
            col_name = k if m.model_params[k][0] == '' else m.model_params[k][0]
            m_stats_df[col_name] = [val] * len(m_stats_df)
        
        if self.aggstats is None:
            self.aggstats = m_stats_df
        else:
            self.aggstats = pd.concat((self.aggstats, m_stats_df), ignore_index=True)

    def fetch_all(self):
        return self.aggstats


def evaluate_model(paths, params):
    df_sh = pd.read_csv(os.path.join(paths['results'], 'synth.csv'))  # load synthetic histories
    df_hist = pd.read_csv(os.path.join(os.path.split(paths['data'])[0], 'Data_0.csv'))
    
    sh_cols = list(df_sh.columns)
    cols_to_drop = []
    for hist_col in df_hist.columns:
        if hist_col not in sh_cols:
            cols_to_drop.append(hist_col)
    df_hist = df_hist.drop(columns=cols_to_drop)

    s = ModelAssessment(df_hist, df_sh, params)
    return s
