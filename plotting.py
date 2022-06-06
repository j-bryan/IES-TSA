import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import dill as pickle


def ercot_wasserstein_heatmaps():
    stats = pd.read_csv('all_stats.csv', index_col=0)

    stats.sort_values(by=['P', 'Q', 'n_clusters', 'pivotLength'], inplace=True)
    stats['preserveInputCDF'] = [True, True, True, False, False, False] * (len(stats) // 6)

    n_clusters_vals = sorted(stats['n_clusters'].unique())
    pivotLength_vals = sorted(stats['pivotLength'].unique(), reverse=True)

    P = np.array(sorted(stats['P'].unique()))
    Q = np.array(sorted(stats['Q'].unique()))[:-1]

    for v in ['WIND', 'SOLAR', 'TOTALLOAD']:
        fig, axes = plt.subplots(nrows=len(Q), ncols=len(P))
        fig.suptitle(v)

        var_df = stats.loc[stats['var'] == v]
        max_wd = max(var_df['WassersteinDist'])
        
        for i, p in enumerate(P):
            for j, q in enumerate(Q):
                # axes[j, i].scatter(dfq['n_clusters'], dfq['pivotLength'], c=dfq['WassersteinDist'] / max_wd)
                # axes[j, i].imshow(dfq[['n_clusters', 'pivotLength']])
                df_rawcdf = var_df.query(f'P == {p} & Q == {q} & preserveInputCDF == False')
                w = build_matrix(df_rawcdf, xcol='n_clusters', x_vals=n_clusters_vals, ycol='pivotLength', y_vals=pivotLength_vals, target='WassersteinDist')
                cmap = cm.get_cmap('plasma_r')
                cmap.set_under('black')
                axes[j, i].imshow(w / max_wd, cmap=cmap, vmin=0, vmax=1)

                axes[j, i].set_xticks(np.arange(len(n_clusters_vals)))
                axes[j, i].set_xticklabels(np.array(n_clusters_vals, dtype=int))
                axes[j, i].set_yticks(np.arange(len(pivotLength_vals)))
                axes[j, i].set_yticklabels(np.array(pivotLength_vals, dtype=int))
                if j == 0:
                    axes[j, i].set_title(f'P={int(p)}')
                if i == 0:
                    axes[j, i].set_ylabel(f'Q={int(q)}')
        plt.savefig(f'./plots/ERCOT/{v}_wasserstein_heatmaps.png', dpi=150)


def build_matrix(df, xcol, x_vals, ycol, y_vals, target):
    X, Y = np.meshgrid(x_vals, y_vals)
    D = -1 * np.ones(X.shape)

    target_col_index = np.argwhere(df.columns == target).ravel()[0]
    
    for i in range(len(X)):
        for j in range(len(X[0])):
            try:
                D[j, i] = df.loc[(df[xcol] == X[j, i]) & (df[ycol] == Y[j, i])].iat[0, target_col_index]  # is there no better way to do this??
            except IndexError:  # not all (n_clusters, pivotLength) pairs had a convergent model, so we'll skip these
                continue
    
    return D


def show_figs(params):
    """ Shows the violin plot and Q-Q plot for a given set of parameters """
    with open('ensemble.pk', 'rb') as f:
        ens = pickle.load(f)
    
    query_str = ' & '.join([f'{k} == {v}' for k, v in params.items()])
    df = ens.aggstats.query(query_str)

    fig_inds = min(df.index) // 3 * 2 + 1
    fig_inds = np.arange(fig_inds, fig_inds + 4)
    
    for fignum in (set(plt.get_fignums()) - set(fig_inds)):
        plt.close(fignum)
    
    plt.show()
    

if __name__ == '__main__':
    # ercot_wasserstein_heatmaps()
    show_figs({'pivotLength': 730, 'P': 1, 'Q': 2, 'n_clusters': 8})
