#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for testing Faster R-CNN model
"""

from __future__ import division

import argparse
import os
import pickle
import time

import numpy as np
from tensorflow import keras as keras

K = keras.backend
Input = keras.layers.Input
Model = keras.models.Model
Progbar = keras.utils.Progbar
import cv2
from keras_frcnn.networks import get_available_networks
from keras_frcnn.utils.logger import rootLogger
import keras_frcnn.roi.roi_helpers as roi_helpers

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

# network-related args
parser.add_argument('-a', '--architecture', type=str, default='resnet',
                    help="architecture name, {'" + \
                         "', '".join(get_available_networks()) + \
                         "'}")
parser.add_argument('-model_weight_path', '--model_weight_path', default='model_frcnn.hdf5',
                    help="Model weight path. If not specified, will give error.")

# optimization-related args
parser.add_argument('-opt', '--optim', type=str, default='adam',
                    help="optimizer, {'" + \
                         "', '".join(OPTIMIZERS.keys()) + \
                         "'}")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')

# Faster R-CNN related args
parser.add_argument('-n', '--num_rois', type=int, default=512,
                    help='Number of RoIs to process at once.')
parser.add_argument("-config_filename", "--config_filename", type=str, default="config.pickle",
                    help="Location to read the metadata related to the training (generated when training).")

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

# print arguments
rootLogger.info("Running with the following parameters:")
rootLogger.info(vars(args))

# which architecture to use {VGG16, Resnet50}
if args.architecture == 'vgg':
    from keras_frcnn.networks import vgg as nn
elif args.architecture == 'resnet':
    from keras_frcnn.networks import resnet as nn
else:
    rootLogger.info('Not a valid model')
    raise ValueError

if not args.model_weight_path:  # if filename is not given
    rootLogger.info('Error: path to test data must be specified. Pass --model_weight_path to command line')

config_output_filename = os.path.join(os.getcwd(), args.config_filename)

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = args.data_dirpath + args.dataset + "/test/"
save_path = args.data_dirpath + args.dataset + "/rcnn_out/"

def format_img_size(img, C):
    """
    Formats the image size based on config
    :param img:
    :param C:
    :return:
    """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """
    Formats the image channels based on config
    :param img:
    :param C:
    :return:
    """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping
print(class_mapping)
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(args.num_rois)

if C.network == 'resnet':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

rootLogger.info('Loading weights from {}'.format(os.path.join(os.getcwd() + '/keras_frcnn/weights/', C.model_path)))
model_rpn.load_weights(os.path.join(os.getcwd() + '/keras_frcnn/weights/', C.model_path), by_name=True)
rootLogger.info("RPN Weights Loaded")
model_classifier.load_weights(os.path.join(os.getcwd() + '/keras_frcnn/weights/', C.model_path), by_name=True)
rootLogger.info("Classifier Weights Loaded")

optimizer_rpn = OPTIMIZERS.get(args.optim, None)(lr=args.learning_rate)
optimizer_classifier = OPTIMIZERS.get(args.optim, None)(lr=args.learning_rate)
model_rpn.compile(optimizer=optimizer_rpn, loss='mse')
model_classifier.compile(optimizer=optimizer_classifier, loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.6

visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    rootLogger.info(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    # if K.image_dim_ordering() == 'tf':
    X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, 'tf', overlap_thresh=0.6)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 4)

            textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            textOrg = (real_x1, real_y1 - 0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 4)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    rootLogger.info('Elapsed time = {}'.format(time.time() - st))
    rootLogger.info(all_dets)
    cv2.imwrite(save_path+img_name,img)
