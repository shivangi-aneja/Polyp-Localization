"""
 init class for all the autoencoders for all the datasets
"""

from object_detection.networks.base import BaseNetwork
from object_detection.networks.fcn8s import *



NETWORKS = {"fcn8s1","fcn8s2"}

def get_available_networks():
    """
    lists all the available networks
    :return: None
    """
    return sorted(NETWORKS)


def make_network(name, *args, **kwargs):
    """
    creates the networks based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in NETWORKS:
        raise ValueError("Invalid network architecture: '{0}'".format(name))

    elif name == "fcn8s1":
        return FCN8s1(*args, **kwargs)

    elif name == "fcn8s2":
        return FCN8s2(*args, **kwargs)
