import sys
import os
import argparse
import dill as pickle
import itertools
import numpy as np
import pandas as pd

from model_fit import write_train_xml, fit_arma_rom
from model_eval import evaluate_model, StatEnsemble

import matplotlib.pyplot as plt
from matplotlib import cm



def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_paths(iso, working_directory):
    DATA_DIR = os.path.join(working_directory, f'data/{iso}_5year')
    # TEMPLATE_PATH = os.path.join(working_directory, f'train_templates/train_template_{iso.lower()}.xml')
    TEMPLATE_PATH = os.path.join(working_directory, f'train_templates_price/train_template_{iso.lower()}.xml')
    RESULTS_BASE_DIR = os.path.join(working_directory, f'model_fit_results/{iso}_2021')
    paths = {
        'base': working_directory,
        'data': DATA_DIR,
        'template': TEMPLATE_PATH,
    }
    return paths


def main(ISO):
    import warnings
    warnings.filterwarnings('ignore')

    BASE_DIR = os.getcwd()
    paths = get_paths(ISO, BASE_DIR)

    # P = [1]
    # Q = [0, 1]
    # L = [24]  # segment lengths
    # K = [4]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    # preserveCDF = ['False']
    P = [1, 2, 3]
    Q = [0, 1, 2, 3]
    # L = [24, 40, 60, 73, 120]  # segment lengths
    L = [24, 48]  # segment lengths
    K = [4, 8, 16, 32]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    preserveCDF = ['True']

    model_params = {'WorkingDir': [('', paths.get('data'))],  # node name: (attribute name, value to set); attribute name '' indicates that it's a node text value
                    'P': [('', str(p)) for p in P],
                    'Q': [('', str(q)) for q in Q],
                    'n_clusters': [('', str(k)) for k in K],
                    'subspace': [('pivotLength', str(l)) for l in L],
                    'preserveInputCDF': [('', s) for s in preserveCDF]}

    ens = StatEnsemble()

    for params in product_dict(**model_params):
        print(params)
        sys.argv = ['main.py']

        p = int(params['P'][1])
        q = int(params['Q'][1])
        pivotLength  = int(params['subspace'][1])
        if p + q + 2 >= pivotLength / 3:  # try to catch cases of too few data for the number of ARMA parameters
            continue

        results_dir = os.path.join(paths['data'],  f'K{params["n_clusters"][1]}_L{params["subspace"][1]}_P{params["P"][1]}_Q{params["Q"][1]}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        paths['results'] = results_dir
        params['WorkingDir'] = ('', paths['results'])

        train_xml_path = write_train_xml(params, paths)
        os.chdir(BASE_DIR)
        try:
            fit_arma_rom(train_xml_path)
        except Exception as e:
            os.chdir(BASE_DIR)
            continue  # just skip to the next one; check log to see why this didn't run correctly
        os.chdir(BASE_DIR)

        res = evaluate_model(paths, params)
        ens.append(res)
        break

    stats_summary = ens.fetch_all()
    stats_summary.to_csv(os.path.join(paths['data'], 'statistics.csv'))


if __name__ == '__main__':
    # main('CAISO')
    # main('ERCOT')
    main('MISO')
