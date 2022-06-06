import argparse
import dill as pickle
import itertools
import numpy as np
import pandas as pd

from tsmodel import TSModel
from model_fit import write_train_xml, fit_arma_rom
from model_eval import evaluate_model, StatEnsemble

import matplotlib.pyplot as plt
from matplotlib import cm

# scipy.stats.ks_2samp(): two-sample Kolmogorov-Smirnov test
# statsmodels.stats.diagnostic.het_breushpagan(): Breush-Pagan test for heteroskedasticity
# scipy.stats.wasserstein_distance(): Wasserstein distance between two 


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def main(working_dir):
    P = [1, 2, 3]
    Q = [0, 1, 2, 3]
    # L = [24]  # segment lengths
    # K = [2]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    L = [24, 146, 365, 730, 2190]  # segment lengths
    K = [2, 4, 8, 16, 32]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    preserveCDF = ['True', 'False']

    model_params = {'WorkingDir': [('', working_dir)],  # node name: (attribute name, value to set); attribute name '' indicates that it's a node text value
                    'P': [('', str(p)) for p in P],
                    'Q': [('', str(q)) for q in Q],
                    'n_clusters': [('', str(k)) for k in K],
                    'subspace': [('pivotLength', str(l)) for l in L],
                    'preserveInputCDF': [('', s) for s in preserveCDF]}

    ens = StatEnsemble()

    for params in product_dict(**model_params):
        n_segments = 8760 / int(params['subspace'][1])
        print('# Segments:', int(n_segments))
        if int(params['n_clusters'][1]) > n_segments:
            continue

        train_xml_path = write_train_xml(params)
        try:
            fit_arma_rom(train_xml_path)
        except ValueError:
            continue  # just skip to the next one; check log to see why this didn't run correctly

        res = evaluate_model(working_dir, params)
        ens.append(res)

    print(ens.fetch_all())
    with open('ensemble.pk', 'wb') as f:
        pickle.dump(ens, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'DIR', help='Data working directory'
    )
    args = parser.parse_args()
    
    main(args.DIR)
