#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for copying one model into another
"""

import argparse
import os
import os.path as osp

import torch

from object_detection.networks import (get_available_networks, make_network)
from object_detection.utils import (get_available_datasets)

# Datasets
DATASETS = {'polyps'}

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Copy Model Weights')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps',
                    help="dataset, {'" + \
                         "', '".join(get_available_datasets()) + \
                         "'}")

parser.add_argument('-a', '--architecture', type=str, default='fcn8s1',
                    help="architecture architecture name, {'" + \
                         "', '".join(get_available_networks()) + \
                         "'}")

parser.add_argument('-sm', '--source_model', type=str, default='fcn8s_1',
                    help="Source Model Name")

parser.add_argument('-dm', '--dest_model', type=str, default='fcn8s_2_512',
                    help="Destination Model Name")

parser.add_argument('-dp', '--dropout', type=float, default=0,
                    help='dropout')

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())


def main(args=args):
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """
    # pylint: disable=line-too-long

    dataset_name = args.dataset
    architecture = args.architecture

    SOURCE_MODEL_PATH = os.path.join(os.getcwd(), 'models/' + dataset_name + '/' + architecture + '/')
    DEST_MODEL_PATH = os.path.join(os.getcwd(), 'models/' + dataset_name + '/' + architecture + '/')

    # build segmentation model
    model = make_network(name=architecture, dropout=args.dropout, n_classes=2)

    checkpoint = torch.load(SOURCE_MODEL_PATH + args.source_model + '_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = 0
    best_iou = 0
    model_params = filter(lambda x: x.requires_grad, model.parameters())
    model_optim = torch.optim.Adam(model_params)
    model_optim.load_state_dict(state_dict=checkpoint['optim_state_dict'])

    torch.save({
        'epoch': start_epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': model_optim.state_dict(),
        'best_mean_iu': best_iou,
    }, osp.join(DEST_MODEL_PATH, args.dest_model + '_checkpoint.pth.tar'))


if __name__ == '__main__':
    main()
