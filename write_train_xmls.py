import os
import subprocess
import argparse
import xml.etree.ElementTree as ET


RAVEN_PATH = 'X:/raven/raven_framework'


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


def run_raven(filepath):
    # print(filepath)
    # print(RAVEN_PATH)
    # print(RAVEN_PATH + ' ' + filepath)
    # os.system(' ' + filepath)
    subprocess.run('bash.exe {} {}'.format(RAVEN_PATH, filepath), shell=True)


def main(working_dir):
    N = 10
    P = 1
    Q = 0
    ptr_csv = ''

    L = [24, 146, 365, 730, 2190]  # segment lengths
    K = [1, 2, 4, 8, 16, 32]  # number of clusters; skip k if there would be fewer than 10 segments per cluster on average
    # We get a total of 12 (L, K) combinations

    for l in L:
        for k in K:
            if (8760 / l) // k < 10:
                continue
            fpath = write_train_xml(working_dir, l, k, P, Q)
            run_raven(fpath)
            break
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'DIR', help='Data working directory'
    )
    args = parser.parse_args()
    
    main(args.DIR)
