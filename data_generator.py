#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for creating .npy files for image
"""


from __future__ import division

import os
import numpy as np
from PIL import Image
import cv2
from skimage import color
from skimage import io


# Resize png images to be of size 512*512
def resize_images(input_path_x, input_path_y):

    new_size = (512,512)

    # For images
    for image in sorted(os.listdir(input_path_x)):
        if image.endswith(".png"):
            img = cv2.imread(input_path_x+image)
            new_image = cv2.resize(img, new_size)
            new_image = cv2.cvtColor(new_image, cv2.IMREAD_COLOR)
            cv2.waitKey(0)
            cv2.imwrite(input_path_x+image.split(".")[0]+'.png',new_image)

    # For labels
    for image in sorted(os.listdir(input_path_y)):
        if image.endswith(".png"):
            img = cv2.imread(input_path_y + image)
            new_image = cv2.resize(img, new_size)
            new_image = cv2.cvtColor(new_image, cv2.IMREAD_COLOR)
            cv2.waitKey(0)
            cv2.imwrite(input_path_y + image.split(".")[0] + '.png', new_image)

def save_to_numpy_array(input_path_x,input_path_y, output_path_x, output_path_y, mode):

    image_list = []
    mask_list = []

    # For renaming the file
    # for image in sorted(os.listdir(input_path_x)):
    #     if image.endswith(".PNG"):
    #         os.rename(input_path_x+image,input_path_x+image.split(".")[0]+'.png')
    #
    # for image in sorted(os.listdir(input_path_y)):
    #     if image.endswith(".PNG"):
    #         os.rename(input_path_y+image,input_path_y+image.split(".")[0]+'.png')

    # For images
    for image in sorted(os.listdir(input_path_x)):

        if image.endswith(".png"):
            image = Image.open(input_path_x+image)
            # This data has shape (height, width, channels)
            data = np.array(image, dtype='uint8')
            # Change to (channels, height, width)
            data = np.transpose(data, [2,0,1])
            image_list.append(data)

    name_x = mode+'_X.npy'
    np.save(os.path.join(output_path_x + name_x), image_list)

    # For labels
    for image in sorted(os.listdir(input_path_y)):

        if image.endswith(".png"):
            # Change the RGB label to Greyscale
            image = color.rgb2gray(io.imread(input_path_y+image))
            # This data has shape (height, width)
            data = np.array(image, dtype='uint8')
            mask_list.append(data)
    name_y = mode + '_Y.npy'
    np.save(os.path.join(output_path_y + name_y), mask_list)


def main():

    input_dir_train_X = os.path.join(os.getcwd(), 'data/polyps/train/image/')
    input_dir_train_y = os.path.join(os.getcwd(), 'data/polyps/train/mask/')

    output_dir_train_X = os.path.join(os.getcwd(), 'data/polyps/')
    output_dir_train_y = os.path.join(os.getcwd(), 'data/polyps/')

    input_dir_test_X = os.path.join(os.getcwd(), 'data/polyps/test/image/')
    input_dir_test_y = os.path.join(os.getcwd(), 'data/polyps/test/mask/')

    output_dir_test_X = os.path.join(os.getcwd(), 'data/polyps/')
    output_dir_test_y = os.path.join(os.getcwd(), 'data/polyps/')

    #resize images
    resize_images(input_path_x = input_dir_train_X, input_path_y=input_dir_train_y)
    resize_images(input_path_x = input_dir_test_X, input_path_y=input_dir_test_y)

    # Train
    save_to_numpy_array(input_path_x=input_dir_train_X, input_path_y=input_dir_train_y,
                        output_path_x=output_dir_train_X, output_path_y=output_dir_train_y, mode='train')

    # Test
    save_to_numpy_array(input_path_x=input_dir_test_X, input_path_y=input_dir_test_y,
                        output_path_x=output_dir_test_X, output_path_y=output_dir_test_y, mode ='test')


if __name__ == '__main__':
    main()