#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for bounding box detection of coordinates
"""


import os
import argparse
import pprint
from PIL import Image
from object_detection.logging.logger import rootLogger
from object_detection.utils import (get_available_datasets)
from scipy import ndimage as ndi
import csv
import cv2
import scipy.misc
import numpy as np


"""" function that finds polyp location from images """


def hotspots_from_images(image):
    hotspots = []
    # Threshold
    thres = 0
    image[image <= thres] = 0
    # Set labels
    labels = ndi.label(image)
    # iterate through labels and find hot spots
    for polyp in range(1, labels[1]+1):
        # Find pixels with each polyp label value
        nonzero = (labels[0] == polyp).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        hotspots.append(bbox)
    # print(hotspots)
    return hotspots


"""" Function to draw bounding box in the image"""


def draw_boxes(img, bboxes, color=(0, 255, 0), thick=1):
    # Make a copy of the image
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        # print(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img


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
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')


# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())


def main(args=args):
    """

    main to find multiple bounding boxes if present and plots them.

    """
    # pylint: disable=line-too-long


    dataset_name = args.dataset

    FILE_PATH = os.path.join(os.getcwd(), 'data/'+ args.dataset+'/val_predictions/')
    file_name = 'bbox_predictions.csv'
    img_list = list()
    final_box = list()
    out_path = os.path.join(os.getcwd(), 'data/'+ args.dataset+'/val_predictions/boxes/')

    for image in sorted(os.listdir(FILE_PATH)):
        if image.endswith("png"):
            img = Image.open(FILE_PATH+image)
            imgarr = np.array(img)
            width, height = img.size
            box = hotspots_from_images(imgarr)
            if len(box) > 1:
                for eachbox in box:
                    print(eachbox)
                    img_list.append(np.array([image,str(width),str(height), eachbox[0][0], eachbox[0][1], eachbox[1][0], eachbox[1][1]]))
                    final_box.append([(eachbox[0][0], eachbox[0][1]), (eachbox[1][0], eachbox[1][1])])
                # print(final_box)
                labelled = draw_boxes(imgarr, final_box)
                scipy.misc.imsave((out_path + image), labelled)
                final_box = list()
            else:
                img_list.append(np.array([image, str(width), str(height), box[0][0][0], box[0][0][1], box[0][1][0], box[0][1][1]]))
                draw_boxes(imgarr, box)
                print(box)
                labelled = draw_boxes(imgarr, box)
                scipy.misc.imsave((out_path + image), labelled)

    with open(FILE_PATH+file_name, mode='w') as bbox_file:
        bbox_writer = csv.writer(bbox_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        bbox_writer.writerow(['Filename', 'Width', 'Height', 'xmin', 'ymin', 'xmax', 'ymax'])

        for i in range(len(img_list)):
            bbox_writer.writerow(img_list[i])


if __name__ == '__main__':
    main()