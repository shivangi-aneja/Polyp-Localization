#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for training Faster R-CNN model
"""

from __future__ import division

import argparse
import os
import pickle
import random
import time
from itertools import cycle

import numpy as np
from tensorflow import keras as keras

K = keras.backend
Input = keras.layers.Input
Model = keras.models.Model
Progbar = keras.utils.Progbar
from keras_frcnn.utils import config, data_generators
from keras_frcnn.utils.data_parser import get_data
from keras_frcnn.networks import get_available_networks
from keras_frcnn.losses import losses as losses
from keras_frcnn.utils.logger import rootLogger
import keras_frcnn.roi.roi_helpers as roi_helpers
from object_detection.logging.tf_logger import Logger

# Optimizers
OPTIMIZERS = {
    'adam': keras.optimizers.Adam,
    'adagrad': keras.optimizers.Adagrad,
    'sgd': keras.optimizers.SGD,
    'rms_prop': keras.optimizers.RMSprop
}

# Datasets
DATASETS = {'polyps_rcnn'}

# General Paths
LOG_PATH = os.path.join(os.getcwd(), 'faster_rcnn_logs/')
TF_LOG_PATH = os.path.join(os.getcwd(), 'faster_rcnn_tf_logs/')
MODEL_PATH = os.path.join(os.getcwd(), 'faster_rcnn_models/')

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Polyp Detection Using Faster R-CNN Network')

# general args
parser.add_argument('-d', '--dataset', type=str, default='polyps_rcnn',
                    help="dataset, {'" + \
                         "', '".join(sorted(DATASETS)) + \
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')
parser.add_argument('--n-workers', type=int, default=4,
                    help='how many threads to use for I/O')
parser.add_argument('-gpu', '--gpu', type=int, default=0,
                    help="ID of the GPU to train on (or -1 to train on CPU)")
parser.add_argument('-rs', '--random-seed', type=int, default=1,
                    help="random seed for training")

# network-related args
parser.add_argument('-a', '--architecture', type=str, default='resnet',
                    help="architecture name, {'" + \
                         "', '".join(get_available_networks()) + \
                         "'}")
parser.add_argument('-b', '--batch_size', type=int, default=2,
                    help='input batch size for training')
parser.add_argument('-e', '--epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('-m', '--model_name', type=str, default='test',
                    help="model name to save")
parser.add_argument('-input_weight_path', '--input_weight_path',
                    help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_argument("-output_weight_path", "--output_weight_path", help="Output path for weights.",
                    default='model_frcnn.hdf5')

# visualization-related args
parser.add_argument('-tf', '--tf_logs', type=str, default='faster_rcnn_tf_logs',
                    help="log folder for tensorflow logging")

# optimization-related args
parser.add_argument('-opt', '--optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) + \
                         "'}")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('-lr_decay', '--lr_decay',
                    help='step to do learning rate decay, unit is epoch',
                    default=5, type=int)
parser.add_argument('-lr_decay_gamma', '--lr_decay_gamma',
                    help='learning rate decay ratio',
                    default=0.1, type=float)
parser.add_argument('-wd', '--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('-dp', '--dropout', type=float, default=0,
                    help='dropout')

# Faster R-CNN related args
parser.add_argument('-n', '--num_rois', type=int, default=32,
                    help='Number of RoIs to process at once.')
parser.add_argument("-hf", "--horizontal_flips",
                    help="Augment with horizontal flips in training. (Default=false).", action="store_true",
                    default=True)
parser.add_argument("-vf", "--vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                    action="store_true", default=True)
parser.add_argument("-rot", "--rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                    action="store_true", default=True)
parser.add_argument("-config_filename", "--config_filename", type=str, default="config.pickle",
                    help="Location to store all the metadata related to the training (to be used when testing).")

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu > -1 else '-1'

# print arguments
rootLogger.info("Running with the following parameters:")
rootLogger.info(vars(args))

# pass the settings from the command line, and persist them in the config object
C = config.Config()

# Use this augmentation only in case of training data, not the validation data
C.use_horizontal_flips = bool(args.horizontal_flips)
C.use_vertical_flips = bool(args.vertical_flips)
C.rot_90 = bool(args.rot_90)

C.model_path = args.output_weight_path
C.num_rois = int(args.num_rois)

# which architecture to use {VGG16, Resnet50}
if args.architecture == 'vgg':
    C.network = 'vgg'
    from keras_frcnn.networks import vgg as nn
elif args.architecture == 'resnet':
    C.network = 'resnet'
    from keras_frcnn.networks import resnet as nn
else:
    rootLogger.info('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if args.input_weight_path:
    rootLogger.info("Loading weights from saved Polyp model")
    C.base_net_weights = os.path.join(os.getcwd() + '/keras_frcnn/weights/', args.input_weight_path)
else:
    # set the path to weights based on backend and model
    rootLogger.info("Loading Pretrained Imagenet weights")
    C.base_net_weights = os.path.join(os.getcwd() + '/keras_frcnn/weights/', nn.get_weight_path())

# Load the data corresponding to training and validation images from csv file
train_imgs, val_imgs, classes_count, class_mapping = get_data(args.data_dirpath + args.dataset)

# Add the background class if not present
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

rootLogger.info('Training images per class:')
rootLogger.info(classes_count)
rootLogger.info('Num classes (including bg) = {}'.format(len(classes_count)))

# Get the config file
config_output_filename = os.path.join(os.getcwd(), args.config_filename)

# Save the config file
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    rootLogger.info('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

rootLogger.info('Num train samples {}'.format(len(train_imgs)))
rootLogger.info('Num val samples {}'.format(len(val_imgs)))

# Get the anchor related data
# Manually generate the regions to be evaluated on RPN network
# data_gen_train = data_generators.get_anchor_gt(img_data=train_imgs, C=C, img_length_calc_function=nn.get_img_output_length, mode='train')
# data_gen_val = data_generators.get_anchor_gt(img_data=val_imgs, C=C, img_length_calc_function=nn.get_img_output_length, mode='val')

# Image Shape for the model
input_shape_img = (None, None, 3)

# Placeholders for keras model
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (VGG here, can be resnet, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# RPN gives the object proposals
rpn = nn.rpn(shared_layers, num_anchors)

# Classifier Network : Gives output class and regression coordinates
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

# Create 2 models for training
# 1. RPN : for getting region proposals
model_rpn = Model(img_input, rpn[:2])
# 2. Classifier : for getting the classifier output
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    rootLogger.info('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    rootLogger.info(
        'Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = OPTIMIZERS.get(args.optim, None)(lr=args.learning_rate)
optimizer_classifier = OPTIMIZERS.get(args.optim, None)(lr=args.learning_rate)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 50
num_epochs = int(args.epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

# Tensorboard Logging
logger = Logger(model_name=C.model_path, data_name=args.dataset, log_path=TF_LOG_PATH)

rootLogger.info('Starting training')

vis = True

# randomly shuffle the data
np.random.shuffle(train_imgs)
train_imgs_cycle = cycle(train_imgs)

for epoch_num in range(num_epochs):
    progbar = Progbar(epoch_length)
    rootLogger.info('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                rootLogger.info(
                    'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                        mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    rootLogger.info(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            train_data = next(train_imgs_cycle)
            print(train_data)

            # X = Original Image, Y = [ Ground Truth Anchor Class (pos, neg, neutral) and Coordinates for Augmented Image ], img_data = Augmented Image
            X, Y, img_data = data_generators.get_anchor_gt(img=train_data, C=C,
                                                           img_length_calc_function=nn.get_img_output_length,
                                                           mode='train')

            loss_rpn = model_rpn.train_on_batch(X, Y)

            # Make predictions on the trained Model
            P_rpn = model_rpn.predict_on_batch(X)

            # P_rpn = Classification, Regression
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=2000)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            # print(len(pos_samples),len(neg_samples))
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            print("ps", len(pos_samples), len(neg_samples))

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]  # RPN Classification Loss (+ve, -ve , neutral)
            losses[iter_num, 1] = loss_rpn[2]  # RPN Regression Loss

            losses[iter_num, 2] = loss_class[1]  # Class Classification Loss
            losses[iter_num, 3] = loss_class[2]  # Class Regression Loss
            losses[iter_num, 4] = loss_class[3]  # Classifier accuracy for bounding boxes from RPN

            iter_num += 1

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                # Log the training losses after every epoch
                logger.log_frcnn(mode='train', rpn_cls=loss_rpn_cls,
                                 rpn_regr=loss_rpn_regr,
                                 detector_cls=loss_class_cls, detector_regr=loss_class_regr,
                                 accuracy=class_acc, step=epoch_num + 1)

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    rootLogger.info('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    rootLogger.info('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    rootLogger.info('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    rootLogger.info('Loss RPN regression: {}'.format(loss_rpn_regr))
                    rootLogger.info('Loss Detector classifier: {}'.format(loss_class_cls))
                    rootLogger.info('Loss Detector regression: {}'.format(loss_class_regr))
                    rootLogger.info('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        rootLogger.info(
                            'Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                model_all.save_weights(os.path.join(os.getcwd() + '/keras_frcnn/weights/', C.model_path))

                break

        except Exception as e:
            rootLogger.info('Exception in Training : {}'.format(e))
            continue

rootLogger.info('Training complete, exiting.')
