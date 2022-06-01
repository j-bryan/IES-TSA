import os
import copy
import xml.etree.ElementTree as ET
from scipy.stats import ks_2samp, wasserstein_distance, describe
from statsmodels.stats.diagnostic import het_breuschpagan


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

    def fit(self):
        """ Use RAVEN to train an ARMA ROM """
        # Make a copy of the template XML tree
        tree = copy.deepcopy(self._tree)

        # Edit template values as needed
        ## segment length
        subspace_node = tree.find('subspace')
        subspace_node.set('pivotLength', str(self._l))
        ## number of clusters
        clusters = tree.find('n_clusters')
        clusters.text = self._k
        ## ARMA P
        arma_p = tree.find('P')
        arma_p.text = self._p
        clusters.text = self._k
        ## ARMA Q
        arma_q = tree.find('Q')
        arma_q.text = self._q

        # Write RAVEN training file
        tree.write('train.xml')

        # Run file using RAVEN
        os.system('raven train.xml')

    def check_fit(self, N=100):
        """ Generates N synthetic histories and calculates a number of statistics """
        # 
        pass

    def summary(self):
        pass
