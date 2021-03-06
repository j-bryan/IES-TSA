import os
import subprocess
import argparse
import xml.etree.ElementTree as ET

import logging
logging.basicConfig(filename='arma_training.log', level=logging.DEBUG)


RAVEN_PATH = 'X:/raven/raven_framework'


def write_train_xml(model_params):
    """ Use RAVEN to train an ARMA ROM """
    # Make a copy of the template XML tree
    template_path = 'train_template.xml'
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Edit template values as needed
    for k, v in model_params.items():
        node = find_node(root, k)
        attr, val = v
        if attr == '':
            node.text = val
        else:
            node.set(attr, val)

    # Write RAVEN training file
    working_dir = find_node(root, 'WorkingDir').text
    filepath = os.path.join(working_dir, 'train.xml')
    tree.write(filepath)

    return filepath


def find_node(root, node):
        for child in root.iter():
            if child.tag == node:
                return child


def fit_arma_rom(filepath):
    subprocess.run('bash.exe {} {}'.format(RAVEN_PATH, filepath), shell=True)
    output = subprocess.run('bash.exe {} {}'.format(RAVEN_PATH, filepath), shell=True, capture_output=True)
    if output.returncode != 0:  # something didn't work right, so we'll manually throw an error
        logging.error(output.stderr)
        raise ValueError
