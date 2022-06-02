import os
import subprocess
import copy
import xml.etree.ElementTree as ET
from scipy.stats import ks_2samp, wasserstein_distance, describe
from statsmodels.stats.diagnostic import het_breuschpagan

import logging
logging.basicConfig(filename='arma_training.log', level=logging.DEBUG)


RAVEN_PATH = 'X:/raven/raven_framework'


class TSModel:
    def __init__(self, L, K, P=1, Q=1):
        self._use_clusters = False if K <= 1 else True
        self._l = L
        self._k = K
        self._p = P
        self._q = Q

        # Load XML template file
        template_path = 'train_template.xml'
        self._tree = ET.parse(template_path)

        self._rom_path = None  # Can I load the pickle into here and call it directly somehow?

    def fit(self, wd):
        """ Use RAVEN to train an ARMA ROM """
        # Make a copy of the template XML tree
        tree = copy.deepcopy(self._tree)
        root = tree.getroot()

        # Edit template values as needed
        ## segment length
        subspace_node = self._find_node(root, 'subspace')
        subspace_node.set('pivotLength', str(self._l))
        ## number of clusters
        clusters = self._find_node(root, 'n_clusters')
        clusters.text = str(self._k)
        ## ARMA P
        arma_p = self._find_node(root, 'P')
        arma_p.text = str(self._p)
        ## ARMA Q
        arma_q = self._find_node(root, 'Q')
        arma_q.text = str(self._q)

        # Write RAVEN training file
        filepath = os.path.join(wd, 'train.xml')
        tree.write(filepath)

        # Run file using RAVEN
        output = subprocess.run('bash.exe {} {}'.format(RAVEN_PATH, filepath), shell=True, capture_output=True)
        if output.returncode != 0:  # something didn't work right, so we'll manually throw an error
            logging.error('L={}, K={}, P={}, Q={}'.format(self._l, self._k, self._p, self._q))
            logging.error(output.stderr)
            raise ValueError
        else:
            logging.info('Success for L={}, K={}, P={}, Q={}'.format(self._l, self._k, self._p, self._q))
    
    def _find_node(self, root, node):
        for child in root.iter():
            if child.tag == node:
                return child

    def check_fit(self, N=100):
        """ Generates N synthetic histories and calculates a number of statistics """
        # 
        pass

    def summary(self):
        pass
