#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for bounding box detection of coordinates
"""

import argparse
import csv
import os
import pprint

import numpy as np
from PIL import Image

from object_detection.logging.logger import rootLogger
from object_detection.utils import (get_available_datasets)

# Datasets
DATASETS = {'polyps'}

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Bounding Box Predictions')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps_rcnn_cvc',
                    help="dataset, {'" + \
                         "', '".join(get_available_datasets()) + \
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

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

    dataset_name = args.dataset

    FILE_PATH = os.path.join(os.getcwd(), 'data/' + args.dataset + '/train/mask/')
    file_name = os.path.join(os.getcwd(), 'data/' + args.dataset + '/train.csv')
    img_list = list()

    for image in sorted(os.listdir(FILE_PATH)):
        if (image.endswith("png")):
            img = Image.open(FILE_PATH + image)
            width, height = img.size
            box = bbox2(np.asarray(img))
            img_list.append(
                np.array([image, str(width), str(height), 'polyp', str(box[0]), str(box[1]), str(box[2]), str(box[3])]))

    with open(file_name, mode='w') as bbox_file:
        bbox_writer = csv.writer(bbox_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        bbox_writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

        for i in range(len(img_list)):
            bbox_writer.writerow(img_list[i])


def bbox2(img):
    a = np.where(img != 0)
    if (len(a[0]) > 0 and len(a[1])):
        bbox = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])
    else:
        bbox = 0, 0, 0, 0
    return bbox


if __name__ == '__main__':
    main()
