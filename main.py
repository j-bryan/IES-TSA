import pandas as pd
from time import time

from tsmodel import TSModel

# scipy.stats.ks_2samp(): two-sample Kolmogorov-Smirnov test
# statsmodels.stats.diagnostic.het_breushpagan(): Breush-Pagan test for heteroskedasticity
# scipy.stats.wasserstein_distance(): Wasserstein distance between two 


def main():
    N = 10
    P = 1
    Q = 1
    ptr_csv = ''

    L = [24, 146, 365, 730, 2190]  # segment lengths
    K = [1, 2, 4, 8, 16, 32]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    # We get a total of 12 (L, K) combinations

    results = pd.DataFrame(columns=['SegLen', 'K_clusters', 'N_samples', 'ARMA_P', 'ARMA_Q', 'KS_stat', 'KS_pvalue',
                                    'BP_lm', 'BP_lm_pvalue', 'BP_fvalue', 'BP_f_pvalue', 'Wasserstein', 'fit_time'])

    for l in L:
        for k in K:
            if (8760 / l) // k < 10:
                continue
            
            start = time()
            model = TSModel(l, k, P, Q)
            model.fit()
            dt_fit = time() - start

            print('Fit Time:', dt_fit)
            exit()

            stats = evaluate_fit(model_loc, N)
            results.loc[len(results.index)] = [l, k, N, P, Q, stats['KS_stat'], stats['KS_pvalue'], stats['BP_lm'], stats['BP_lm_pvalue'],
                                               stats['BP_fvalue'], stats['BP_f_pvalue'], stats['Wasserstein'], dt_fit]
    
    results.to_csv('gap_results.csv')


if __name__ == '__main__':
    main()
