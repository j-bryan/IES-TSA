import sys
import os
import argparse
import dill as pickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from model_fit import write_train_xml, fit_arma_rom
from model_eval import evaluate_model, StatEnsemble

import matplotlib.pyplot as plt
from matplotlib import cm


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def worker(params):
    paths = params.pop('paths')

    results_dir = os.path.join(paths['wdir'],  f'K{params["n_clusters"][1]}_L{params["subspace"][1]}_P{params["P"][1]}_Q{params["Q"][1]}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    paths['results'] = results_dir
    params['WorkingDir'] = ('', paths['results'])

    train_xml_path = write_train_xml(params, paths)
    try:
        fit_arma_rom(train_xml_path)
    except Exception as e:
        return None

    res = evaluate_model(paths, params)

    return res


def main(args):
    import warnings
    warnings.filterwarnings('ignore')

    BASE_DIR = os.getcwd()
    paths = args
    paths['base'] = BASE_DIR
    paths = {k: os.path.abspath(v) for k, v in paths.items()}

    P = [1, 2]
    Q = [0]
    L = [24]  # segment lengths
    K = [4]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    preserveCDF = ['True']
    # P = [1, 2, 3]
    # Q = [0, 1, 2, 3]
    # L = [24, 48]  # segment lengths
    # K = [4, 8, 16, 32]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    # preserveCDF = ['True']

    model_params = {'WorkingDir': [('', paths.get('data'))],  # node name: (attribute name, value to set); attribute name '' indicates that it's a node text value
                    'P': [('', str(p)) for p in P],
                    'Q': [('', str(q)) for q in Q],
                    'n_clusters': [('', str(k)) for k in K],
                    'subspace': [('pivotLength', str(l)) for l in L],
                    'preserveInputCDF': [('', s) for s in preserveCDF]}
    
    all_params = list(product_dict(**model_params))
    param_sets_to_drop = []
    for i, params in enumerate(all_params):
        all_params[i]['paths'] = paths
        p = int(params['P'][1])
        q = int(params['Q'][1])
        pivotLength  = int(params['subspace'][1])
        if p + q + 2 >= pivotLength / 3:  # try to catch cases of too few data for the number of ARMA parameters
            param_sets_to_drop.append(params)
    for params in param_sets_to_drop:
        all_params.remove(params)
    
    with mp.Pool() as p:
        results = p.map(func=worker, iterable=all_params)

    ens = StatEnsemble()
    for res in results:
        ens.append(res)
    stats_summary = ens.fetch_all()
    stats_summary.to_csv(os.path.join(paths['wdir'], 'statistics.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--wdir', required=True, type=str)
    parser.add_argument('-t', '--template', required=True, type=str)
    parser.add_argument('-d', '--data', required=True, type=str)
    args = vars(parser.parse_args())
    
    main(args)
