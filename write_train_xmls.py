import os
import subprocess
import argparse
import xml.etree.ElementTree as ET


def write_train_xml(working_dir, L, K, P, Q):
    """ Use RAVEN to train an ARMA ROM """
    # Make a copy of the template XML tree
    template_path = 'train_template.xml'
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Edit template values as needed
    ## working directory
    wd_node = find_node(root, 'WorkingDir')
    wd_node.text = working_dir
    ## segment length
    subspace_node = find_node(root, 'subspace')
    subspace_node.set('pivotLength', str(L))
    ## number of clusters
    clusters = find_node(root, 'n_clusters')
    clusters.text = str(K)
    ## ARMA P
    arma_p = find_node(root, 'P')
    arma_p.text = str(P)
    ## ARMA Q
    arma_q = find_node(root, 'Q')
    arma_q.text = str(Q)

    # Write RAVEN training file
    filepath = os.path.join(working_dir, 'train.xml')
    tree.write(filepath)

    return filepath


def find_node(root, node):
        for child in root.iter():
            if child.tag == node:
                return child
