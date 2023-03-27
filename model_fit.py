import os
import subprocess
import argparse
import xml.etree.ElementTree as ET
import sys

# import logging
# logging.basicConfig(filename='arma_training.log', level=logging.DEBUG)

# RAVENFRAMEWORK = '/home/force-ies/raven'  # TODO: change this to the correct path for your machine
RAVENFRAMEWORK = '/root/raven'  # TODO: change this to the correct path for your machine
sys.path.append(RAVENFRAMEWORK)

from ravenframework.Driver import main


def write_train_xml(model_params, paths):
    """ Use RAVEN to train an ARMA ROM """
    # Make a copy of the template XML tree
    tree = ET.parse(paths['template'])
    root = tree.getroot()

    # Edit template values as needed
    for k, v in model_params.items():
        node = find_node(root, k)
        attr, val = v
        if attr == '':
            node.text = val
        else:
            node.set(attr, val)
    
    input_data = root.find('.//Input[@name="input"]')
    input_data.text = paths['data']

    filepath = os.path.join(paths['results'], 'train.xml')
    tree.write(filepath)

    return filepath


def find_node(root, node):
    for child in root.iter():
        if child.tag == node:
            return child


def fit_arma_rom(filepath):
    # sys.argv.append(filepath)
    sys.argv = [sys.argv[0], filepath]
    main(False)
