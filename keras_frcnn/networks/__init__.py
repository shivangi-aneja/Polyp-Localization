"""
 init class for all the networks for all the datasets
"""

from keras_frcnn.networks.resnet import *
from keras_frcnn.networks.vgg import *

NETWORKS = {"vgg", "resnet"}


def get_available_networks():
    """
    lists all the available networks
    :return: None
    """
    return sorted(NETWORKS)
