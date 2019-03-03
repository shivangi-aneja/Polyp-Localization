#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Main file to evaluate the models
"""
import argparse
import os
import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from object_detection.localization_training import DeepSegmentation
from object_detection.logging.logger import rootLogger
from object_detection.losses.loss_functions import (CrossEntropy2D, DiceLoss)
from object_detection.networks import (get_available_networks,
                                       make_network)
from object_detection.utils import (get_available_datasets,
                                    make_dataset, RNG)
from object_detection.utils.dataset.pytorch_dataset_utils import DatasetIndexer

# Optimizers
OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'sgd': torch.optim.SGD,
    'rms_prop': torch.optim.RMSprop,
    'lbgfs': torch.optim.LBFGS
}

LOSS_FUNCS = {
    'mse': nn.MSELoss(),
    'cse': CrossEntropy2D(),
    'dse': DiceLoss()
}

# Datasets
DATASETS = {'polyps'}

# General Paths
LOG_PATH = os.path.join(os.getcwd(), 'logs/')
TF_LOG_PATH = os.path.join(os.getcwd(), 'tf_logs/')
MODEL_PATH = os.path.join(os.getcwd(), 'models/')

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Polyp Detection Using FCN-8s')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps',
                    help="dataset, {'" + \
                         "', '".join(get_available_datasets()) + \
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')
parser.add_argument('--n-workers', type=int, default=4,
                    help='how many threads to use for I/O')
parser.add_argument('-gpu', '--gpu', type=int, default=0,
                    help="ID of the GPU to train on (or -1 to train on CPU)")
parser.add_argument('-rs', '--random-seed', type=int, default=1,
                    help="random seed for training")

# network-related
parser.add_argument('-a', '--architecture', type=str, default='fcn8s1',
                    help="architecture architecture name, {'" + \
                         "', '".join(get_available_networks()) + \
                         "'}")
parser.add_argument('-l', '--loss', type=str, default='cse',
                    help="Loss function for training,  {'" + \
                         "', '".join(LOSS_FUNCS.keys()) + \
                         "'}")
parser.add_argument('-b', '--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('-m', '--model_name', type=str, default='test',
                    help="model name to save")

# optimization-related
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')

parser.add_argument('-opt', '--optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) + \
                         "'}")

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu > -1 else '-1'

# print arguments
rootLogger.info("Running with the following parameters:")
pprint.pprint(vars(args))


def main(args=args):
    """
    main function that parses the arguments and trains
    :param args: arguments related
    :return: None
    """
    # pylint: disable=line-too-long

    # get variables
    batch_size = args.batch_size
    lr = args.learning_rate
    architecture = args.architecture
    num_workers = args.n_workers
    gpu_id = args.gpu
    model_name = args.model_name

    # load and shuffle data
    dataset = make_dataset(args.dataset)

    train_dataset, val_dataset, test_dataset = dataset.load(args.data_dirpath)
    n_classes = dataset.n_classes()

    rng = RNG(args.random_seed)
    val_ind = rng.permutation(len(val_dataset))

    val_dataset = DatasetIndexer(val_dataset, val_ind)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    IMAGE_PATH = os.path.join(os.getcwd(), 'data/' + args.dataset + '/val_predictions/')

    # build segmentation model
    model = make_network(name=architecture, dropout=0, n_classes=n_classes)

    # get optimizer
    optim = OPTIMIZERS.get(args.optim, None)
    if not optim:
        raise ValueError("invalid optimizer: '{0}'".format(args.optim))

    # get loss function
    loss_func = LOSS_FUNCS.get(args.loss, None)
    if not loss_func:
        raise ValueError("Invalid loss function: '{0}'".format(args.loss))

    out_path = MODEL_PATH + args.dataset + "/" + architecture + "/"

    checkpoint = torch.load(out_path + model_name + '_model_best.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create Segmentation model according to params
    seg_model = DeepSegmentation(model=model, dataset=args.dataset, model_name=model_name, gpu_id=gpu_id, epochs=0,
                                 optim=optim, batch_size=batch_size, lr=lr, optim_kwargs={'lr': lr},
                                 loss_func=loss_func, tf_log_path=None, log_path=LOG_PATH, out_path=out_path,
                                 image_path=IMAGE_PATH)

    seg_model.optim.load_state_dict(state_dict=checkpoint['optim_state_dict'])

    # Validate the model on the data
    seg_model.validate(val_loader=val_loader, n_classes=n_classes)


if __name__ == '__main__':
    main()
