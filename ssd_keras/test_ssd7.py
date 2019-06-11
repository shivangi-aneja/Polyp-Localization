#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for testing SSD 7 model
"""
from tensorflow import keras

K = keras.backend
Input = keras.layers.Input
Model = keras.models.Model
Progbar = keras.utils.Progbar
Adam = keras.optimizers.Adam
CSVLogger = keras.callbacks.CSVLogger
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
TerminateOnNaN = keras.callbacks.TerminateOnNaN
load_model = keras.models.load_model
import argparse
from misc_utils import config_ssd7 as Config
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import pprint
import numpy as np
from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from eval_utils.average_precision_evaluator import Evaluator
from data_generator.object_detection_2d_data_generator import DataGenerator

# Datasets
DATASETS = {'polyps_rcnn'}

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Create h5py files')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps_hospital',
                    help="dataset, {'" + \
                         "', '".join(sorted(DATASETS)) + \
                         "'}")
parser.add_argument('-b', '--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('-m', '--model_name', type=str, default='ssd7_model',
                    help="model name to save")
parser.add_argument('-tf', '--tf_logs', type=str, default='tf_logs',
                    help="folder for tensorflow logging")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('-gpu', '--gpu', type=int, default=0,
                    help="ID of the GPU to train on (or -1 to train on CPU)")

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu > -1 else '-1'

pprint.pprint(vars(args))


def main():
    model_mode = 'inference'
    K.clear_session()  # Clear previous models from memory.

    model = build_model(image_size=(Config.img_height, Config.img_width, Config.img_channels),
                        n_classes=Config.n_classes, mode=model_mode, l2_regularization=Config.l2_regularization,
                        scales=Config.scales,
                        aspect_ratios_per_layer=Config.steps,
                        two_boxes_for_ar1=True, steps=Config.steps, offsets=Config.offsets, clip_boxes=False,
                        variances=Config.variances, normalize_coords=Config.normalize_coords,
                        subtract_mean=Config.intensity_mean,
                        swap_channels=[2, 1, 0], confidence_thresh=0.01, iou_threshold=0.45, top_k=200,
                        nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    weights_path = os.getcwd() + '/weights/' + args.model_name + ".h5"
    model.load_weights(weights_path, by_name=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    test_dataset = DataGenerator(load_images_into_memory=True,
                                 hdf5_dataset_path=os.getcwd() + "/data/" + args.dataset + '/polyp_test.h5')

    test_dataset_size = test_dataset.get_dataset_size()
    print("Number of images in the test dataset:\t{:>6}".format(test_dataset_size))

    classes = ['background', 'polyp']

    evaluator = Evaluator(model=model, n_classes=Config.n_classes, data_generator=test_dataset, model_mode=model_mode)

    results = evaluator(img_height=Config.img_height, img_width=Config.img_width, batch_size=args.batch_size,
                        data_generator_mode='resize',
                        round_confidences=False, matching_iou_threshold=0.5, border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample', num_recall_points=11, ignore_neutral_boxes=True,
                        return_precisions=True, return_recalls=True, return_average_precisions=True, verbose=True)

    mean_average_precision, average_precisions, precisions, recalls = results

    for i in range(1, len(average_precisions)):
        print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))

    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))

    m = max((Config.n_classes + 1) // 2, 2)
    n = 2

    fig, cells = plt.subplots(m, n, figsize=(n * 8, m * 8))
    val = 0
    for i in range(m):
        for j in range(n):
            if n * i + j + 1 > Config.n_classes: break
            cells[i, j].plot(recalls[n * i + j + 1], precisions[n * i + j + 1], color='blue', linewidth=1.0)
            cells[i, j].set_xlabel('recall', fontsize=14)
            cells[i, j].set_ylabel('precision', fontsize=14)
            cells[i, j].grid(True)
            cells[i, j].set_xticks(np.linspace(0, 1, 11))
            cells[i, j].set_yticks(np.linspace(0, 1, 11))
            cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n * i + j + 1], average_precisions[n * i + j + 1]),
                                  fontsize=16)
            image = plt.gcf()
            # plt.show()
            plt.draw()
            image.savefig(os.getcwd() + "/test_out/test_" + str(val) + ".png", dpi=100)
            val += 1


if __name__ == '__main__':
    main()
