"""
base class for autoencoder creation
"""
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Base Class for autoencoders
    """

    def __init__(self, dropout, n_classes):
        """
        initialize the parameters
        :param dropout: dropout rate for the network
        """
        super(BaseNetwork, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes
        self.network = self.make_network()
        self.init()

    def make_network(self):
        """
        create the encoder part of the autoencoder
        :return: encoder
        """
        raise NotImplementedError('`make_network` is not implemented')

    def init(self):
        """
        init method
        :return:
        """
        pass

    def forward(self, x):
        """
        create the encoder part of the autoencoder
        :return: encoder
        """
        raise NotImplementedError('`forward` is not implemented')
