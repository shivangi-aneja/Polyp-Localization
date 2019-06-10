#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for creating HDF5 dataset
"""

import argparse
import os

from data_generator.object_detection_2d_data_generator import DataGenerator

# Datasets
DATASETS = {'polyps_hospital'}

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Create h5py files')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps_hospital',
                    help="dataset, {'" + \
                         "', '".join(sorted(DATASETS)) + \
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

    # Images
    images_dir = os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/images/"

    # # Ground truth
    train_labels_filename = os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/train.csv"
    val_labels_filename = os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/val.csv"
    test_labels_filename = os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/test.csv"

    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    #
    train_dataset.parse_csv(images_dir=images_dir,
                            labels_filename=train_labels_filename,
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                            # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                            include_classes='all')

    val_dataset.parse_csv(images_dir=images_dir,
                          labels_filename=val_labels_filename,
                          input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                          include_classes='all')

    test_dataset.parse_csv(images_dir=images_dir,
                           labels_filename=test_labels_filename,
                           input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                           include_classes='all')

    # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # want to create HDF5 datasets, comment out the subsequent two function calls.

    train_dataset.create_hdf5_dataset(
        file_path=os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/polyp_train.h5",
        resize=False,
        variable_image_size=True,
        verbose=True, images_dir=images_dir)

    val_dataset.create_hdf5_dataset(
        file_path=os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/polyp_val.h5",
        resize=False,
        variable_image_size=True,
        verbose=True, images_dir=images_dir)

    test_dataset.create_hdf5_dataset(
        file_path=os.path.abspath(os.path.join(os.getcwd(), '')) + "/data/" + args.dataset + "/polyp_test.h5",
        resize=False,
        variable_image_size=True,
        verbose=True, images_dir=images_dir)


if __name__ == '__main__':
    main()
