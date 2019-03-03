#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for converting the csv file into bounding box image
"""

import numpy as np
import os
import argparse
import pprint
from PIL import Image
from object_detection.logging.logger import rootLogger
from object_detection.utils import (get_available_datasets)
import csv
import cv2



# Datasets
DATASETS = {'polyps'}


# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Bounding Box Predictions')


# general
parser.add_argument('-d', '--dataset', type=str, default='polyps',
                    help="dataset, {'" +\
                         "', '".join(get_available_datasets()) +\
                         "'}")


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

    FILE_PATH_SRC = os.path.join(os.getcwd(), 'data/'+ args.dataset+'/val_predictions/')
    FILE_PATH_DEST = os.path.join(os.getcwd(), 'data/'+ args.dataset+'/val_predictions_bbox/')
    file_name = 'bbox_predictions.csv'

    with open(FILE_PATH_SRC+file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                image = row[0]
                xmin  = row[3]
                ymin = row[4]
                xmax = row[5]
                ymax = row[6]
                img = cv2.imread(FILE_PATH_DEST+image)
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                cv2.imwrite(FILE_PATH_DEST+image,img)
                line_count += 1




if __name__ == '__main__':
    main()