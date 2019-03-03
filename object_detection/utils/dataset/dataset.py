"""
    dataset class for all the datasets
"""
import os
import torch
import numpy as np
import pickle
import csv
import random
import cv2
import copy
from os.path import dirname
from keras_frcnn.utils.data_generators import iou
from .custom_dataloader import SegDataset,BBoxDataset


def get_available_datasets():
    """
    gets all the datasets
    :return:
    """
    return sorted(DATASETS)


def make_dataset(name):
    """
    it returns dataset according to its name
    :param name: dataset name
    :return: dataset
    """
    name = name.strip().lower()
    if not name in DATASETS:
        raise ValueError("invalid dataset: '{0}'".format(name))
    elif name == 'polyps':
        return POLYPS()
    elif name == 'polyps_rcnn':
        return POLYPS_RCNN()


class BaseDataset(object):
    """
    base dataset
    """
    def _load(self, dirpath):
        """Download if needed and return the dataset.
        Return format is (`torch.Tensor` data, `torch.Tensor` label) iterable
        of appropriate shape or tuple (train data, test data) of such.
        """
        raise NotImplementedError('`load` is not implemented')

    def load(self, dirpath):
        """
        loads the dataset
        :param dirpath: directory path where dataset is stored
        :return:
        """
        return self._load(os.path.join(dirpath, self.__class__.__name__.lower()))

    def n_classes(self):
        """Get number of classes."""
        raise NotImplementedError('`n_classes` is not implemented')


class POLYPS(BaseDataset):
    """
    POLYPS dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        # Training images
        train_val_x = np.load(dirpath + '/train_X.npy')
        train_val_y = np.load(dirpath + '/train_Y.npy')

        train_x = torch.stack([torch.Tensor(i) for i in train_val_x[0:300]])
        val_x = torch.stack([torch.Tensor(i) for i in train_val_x[300:]])

        train_y = torch.stack([torch.Tensor(i) for i in train_val_y[0:300]])
        val_y = torch.stack([torch.Tensor(i) for i in train_val_y[300:]])

        test_x = torch.stack([torch.Tensor(i) for i in np.load(dirpath + '/test_X.npy')])
        test_y = torch.stack([torch.Tensor(i) for i in np.load(dirpath + '/test_Y.npy')])

        train = SegDataset(train_x, train_y)
        val = SegDataset(val_x, val_y)
        test = SegDataset(test_x,test_y)

        return train, val, test

    def n_classes(self):
        # In segmentation n_classes = 2 (One foreground, other background)
        return 2


class POLYPS_RCNN(BaseDataset):
    """
    POLYPS_RCNN dataset
    """
    def __init__(self):
        super(BaseDataset, self).__init__()

    def _load(self, dirpath):
        '''

        :param dirpath:
        :return:
        '''

        visualise = False

        train_annot_path = os.path.join(dirpath, 'train_aug.csv')
        val_annot_path = os.path.join(dirpath, 'val.csv')

        # Get Image Path
        imgsets_path_train = os.path.join(dirpath, 'train_aug')
        imgsets_path_val = os.path.join(dirpath, 'val')

        train_files = []
        val_files = []
        train_annotation_data = []
        val_annotation_data = []
        try:

            # Training Data
            with open(train_annot_path) as f:
                csv_reader = csv.reader(f, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        train_files.append(row[0])
                        annotation_data = {'filepath': os.path.join(imgsets_path_train, row[0]), 'width': row[1],
                                           'height': row[2], 'class': row[3],
                                           'bboxes': {'x1': row[4], 'x2': row[6], 'y1': row[5], 'y2': row[7]}}
                        train_annotation_data.append(annotation_data)
                        if visualise:
                            img = cv2.imread(annotation_data['filepath'])
                            for bbox in annotation_data['bboxes']:
                                cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                            cv2.imshow('img', img)
                            cv2.waitKey(0)
        except Exception as e:
            print(e)

        try:

            # Validation Data
            with open(val_annot_path) as f:
                csv_reader = csv.reader(f, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        val_files.append(row[0])
                        annotation_data = {'filepath': os.path.join(imgsets_path_val, row[0]), 'width': row[1],
                                           'height': row[2], 'class': row[3],
                                           'bboxes': {'x1': row[4], 'x2': row[6], 'y1': row[5], 'y2': row[7]}}
                        val_annotation_data.append(annotation_data)
                        if visualise:
                            img = cv2.imread(annotation_data['filepath'])
                            for bbox in annotation_data['bboxes']:
                                cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                            cv2.imshow('img', img)
                            cv2.waitKey(0)
        except Exception as e:
            print(e)

        config_file = open(os.path.join(dirname(dirname(dirname(os.getcwd()))), 'faster_rcnn/config.pickle'), "rb")
        C = pickle.load(config_file)
        architecture = C.network
        img_length_calc_function = get_img_output_length_vgg if architecture=='vgg16' else get_img_output_length_resnet
        train = get_anchor_gt(train_annotation_data,img_length_calc_function, C,mode='train')
        val = get_anchor_gt(train_annotation_data, img_length_calc_function, C,mode='val')
        train = BBoxDataset(train)
        val = BBoxDataset(val)

        return train, val

    def n_classes(self):
        # n_classes = 2 (Now : polyp and background) ('hyperplastic', 'tubular_adenoma', 'serrated_adenoma')
        return 2


# Create Data Augmentation
def get_anchor_gt(all_img_data, img_length_calc_function, C, mode='train'):



    for img_data in all_img_data:
        try:

            # read in image, and optionally add augmentation
            if mode == 'train':
                img_data_aug, x_img = augment(img_data, C, augment=True)
            else:
                img_data_aug, x_img = augment(img_data, C, augment=False)

            (width, height) = (img_data_aug['width'], img_data_aug['height'])
            (rows, cols, _) = x_img.shape

            assert cols == width
            assert rows == height

            # get image dimensions for resizing
            (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

            # resize the image so that smalles side is length = 600px
            x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

            try:
                y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
                                                 img_length_calc_function)
            except:
                continue

            # Zero-center by mean pixel, and preprocess image

            x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
            x_img = x_img.astype(np.float32)
            x_img[:, :, 0] -= C.img_channel_mean[0]
            x_img[:, :, 1] -= C.img_channel_mean[1]
            x_img[:, :, 2] -= C.img_channel_mean[2]
            x_img /= C.img_scaling_factor

            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis=0)

            y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

            # if backend == 'tf':
            # 	x_img = np.transpose(x_img, (0, 2, 3, 1))
            # 	y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
            # 	y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

            yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

        except Exception as e:
            print(e)
            continue


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height



# Calculate Region Proposal Networks
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    downscale = float(C.rpn_stride)
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture

    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img


def get_img_output_length_vgg(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)

def get_img_output_length_resnet(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length


DATASETS = {"polyps", "polyps_rcnn"}



