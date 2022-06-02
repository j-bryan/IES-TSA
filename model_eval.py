import os

import numpy as np
import scipy
import pandas as pd
import statsmodels as sm

from statsmodels.graphics.gofplots import qqplot_2samples
from statsmodels.graphics.boxplots import violinplot
import seaborn as sns
import matplotlib.pyplot as plt


class ModelAssessment:
    def __init__(self, df_h, df_s, model_params):
        self.df_h = df_h
        self.df_s = df_s
        self._zero_mask_vars = ['SOLAR']
        self.model_params = model_params
        var_names = df_h.columns[1:]  # drop HOUR column
        self.names = var_names

        data = []
        for n in var_names:
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)

            ks_res = scipy.stats.ks_2samp(vals_h, vals_s)
            wd = scipy.stats.wasserstein_distance(vals_h, vals_s)
            data.append([n, ks_res.statistic, ks_res.pvalue, wd])

        self.stats = pd.DataFrame(data, columns=['var', 'KS2samp_Fstat', 'KS2samp_pvalue', 'WassersteinDist'])

        self.qqplot = self._qqplot()
        self.violinplot = self._violinplot()
    
    def _mask_zeros(self, v, n):
        v_tmp = np.asarray(v)
        if n in self._zero_mask_vars:
            return v_tmp[v_tmp != 0]
        else:
            return v_tmp
    
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
    
    def _qqplot(self):
        fig, ax = plt.subplots(ncols=3)
        fig.suptitle(self._params_title())
        for i, n in enumerate(self.names):
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)
            ax[i].set_title(n)
            ax[i].set_aspect('equal')
            qqplot_2samples(vals_s, vals_h, xlabel='Synthetic', ylabel='Historical', line='45', ax=ax[i])
        return fig

    def _violinplot(self):
        data = []
        fig, ax = plt.subplots()
        fig.suptitle(self._params_title())
        for n in self.names:
            vals_h = self._mask_zeros(self.df_h[n], n)
            vals_s = self._mask_zeros(self.df_s[n], n)
            min_val = min(min(vals_h), min(vals_s))
            max_val = max(max(vals_h), max(vals_s))
            vals_h_scaled = (vals_h - min_val) / (max_val - min_val)
            vals_s_scaled = (vals_s - min_val) / (max_val - min_val)
            data.extend(zip(['Historical'] * len(vals_h_scaled), [n] * len(vals_h_scaled), list(vals_h_scaled)))
            data.extend(zip(['Synthetic'] * len(vals_s_scaled), [n] * len(vals_s_scaled), list(vals_s_scaled)))
        df_tmp = pd.DataFrame(data, columns=['Source', 'Variable', 'Value'])
        ax = sns.violinplot(x="Variable", y="Value", hue="Source", data=df_tmp, split=True)
        ax.set_ylabel('MinMax Scaled Value')
        return fig
    
    def summarize(self):
        return ModelSummary(self)


class ModelSummary:
    """ Class to condense some of the data contained in a ModelAssessment object """
    def __init__(self, m):
        self.stats = m.stats
        self.model_params = m.model_params
        self.qqplot = m.qqplot
        self.violinplot = m.violinplot


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

    def violin_lattice(self, params=None):
        pass

    def qq_lattice(self, params=None):
        pass

    def boxplot(self, params=None):
        pass


def evaluate_model(wd, params):
    df_sh = pd.read_csv(os.path.join(wd, 'synth.csv'))  # load synthetic histories
    df_hist = pd.read_csv(os.path.join(wd, 'Data_0.csv'))
    s = ModelAssessment(df_hist, df_sh, params)
    return s
