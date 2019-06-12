#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for testing SSD 512 model
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
from misc_utils import config_ssd512 as Config
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import pprint
import numpy as np
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from eval_utils.average_precision_evaluator import Evaluator
from data_generator.object_detection_2d_data_generator import DataGenerator

# Datasets
DATASETS = {'polyps_rcnn', 'polyps_hospital'}

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
parser.add_argument('-m', '--model_name', type=str, default='ssd512_best',
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

    model = ssd_512(image_size=(Config.img_height, Config.img_width, Config.img_channels), n_classes=Config.n_classes,
                    mode=model_mode, l2_regularization=Config.l2_regularization,
                    scales=Config.scales,
                    aspect_ratios_per_layer=Config.aspect_ratios,
                    two_boxes_for_ar1=True, steps=Config.steps, offsets=Config.offsets, clip_boxes=False,
                    variances=Config.variances, normalize_coords=Config.normalize_coords,
                    subtract_mean=Config.mean_color,
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

    generator = test_dataset.generate(batch_size=1,
                                      shuffle=True,
                                      transformations=[],
                                      returns={'processed_images',
                                               'filenames',
                                               'inverse_transform',
                                               'original_images',
                                               'original_labels'},
                                      keep_images_without_gt=False)

    # Generate a batch and make predictions.

    i = 0
    confidence_threshold = Config.confidence_threshold

    for val in range(test_dataset_size):
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
            generator)

        # print("Image:", batch_filenames[i])
        print("Ground truth boxes:\n")
        print(np.array(batch_original_labels[i]))

        y_pred = model.predict(batch_images)

        # Perform confidence thresholding.
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

        # Convert the predictions for the original image.
        # y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[i])

        plt.figure(figsize=(20, 12))
        plt.imshow(batch_images[i])

        current_axis = plt.gca()

        colors = plt.cm.hsv(np.linspace(0, 1, Config.n_classes + 1)).tolist()  # Set the colors for the bounding boxes
        classes = ['background', 'polyps']  # Just so we can print class names onto the image instead of IDs

        for box in batch_original_labels[i]:
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white',
                              bbox={'facecolor': 'green', 'alpha': 1.0})

        for box in y_pred_thresh[i]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        image = plt.gcf()
        # plt.show()
        plt.draw()
        image.savefig(os.getcwd() + "/val_ssd512/val_" + str(val) + ".png", dpi=100)

    evaluator = Evaluator(model=model, n_classes=Config.n_classes, data_generator=test_dataset, model_mode=model_mode)

    results = evaluator(img_height=Config.img_height, img_width=Config.img_width, batch_size=args.batch_size,
                        data_generator_mode='resize',
                        round_confidences=False, matching_iou_threshold=0.3, border_pixels='include',
                        sorting_algorithm='quicksort',
                        average_precision_mode='sample', num_recall_points=11, ignore_neutral_boxes=True,
                        return_precisions=True, return_recalls=True, return_average_precisions=True, verbose=True)

    mean_average_precision, average_precisions, precisions, recalls, tp_count, fp_count, fn_count, polyp_precision, polyp_recall = results

    print("TP : %d, FP : %d, FN : %d " % (tp_count, fp_count, fn_count))
    print("{:<14}{:<6}{}".format('polyp', 'Precision', round(polyp_precision, 3)))
    print("{:<14}{:<6}{}".format('polyp', 'Recall', round(polyp_recall, 3)))
    # for i in range(1, len(average_precisions)):
    #     print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    #
    # print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
    # print('Precisions', precisions)
    # print('Recalls', recalls)

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
            image.savefig(os.getcwd() + "/test_out_512/test_" + str(val) + ".png", dpi=100)
            val += 1


if __name__ == '__main__':
    main()
