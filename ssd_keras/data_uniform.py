#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for making csv file uniform for input to the network
"""

import csv
import os

import cv2

train_img = os.getcwd() + "/data/polyps_rcnn/train/"
train_file = os.getcwd() + "/data/polyps_rcnn/train.csv"
train_file_new = os.getcwd() + "/data/polyps_rcnn/train_new.csv"

val_img = os.getcwd() + "/data/polyps_rcnn/val/"
val_file = os.getcwd() + "/data/polyps_rcnn/val.csv"

test_img = os.getcwd() + "/data/polyps_rcnn/test/"
test_file = os.getcwd() + "/data/polyps_rcnn/test.csv"
resized_width = 512
resized_height = 512


def resize_data(file_path_read, file_path_write, img_path):
    new_data = []
    new_data.append(["image_name", "xmin", "xmax", "ymin", "ymax", "class_id", "width", "height"])

    try:
        # Read and resize the data
        with open(file_path_read) as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    # row = row[0].split(";")
                    file_name = row[0]
                    print(file_name)
                    xmin = row[4]
                    xmax = row[6]
                    ymin = row[5]
                    ymax = row[7]
                    class_id = 1  # row[5]
                    width = row[1]
                    height = row[2]
                    xmin = int(int(xmin) * (resized_width / float(width)))
                    xmax = int(int(xmax) * (resized_width / float(width)))
                    ymin = int(int(ymin) * (resized_height / float(height)))
                    ymax = int(int(ymax) * (resized_height / float(height)))

                    img = cv2.imread(img_path + file_name)
                    # cv2.resize(img, (resized_width, resized_height))
                    cv2.imwrite(img_path + file_name, img)
                    new_data.append(
                        [file_name, str(xmin), str(xmax), str(ymin), str(ymax), str(class_id), str(resized_width),
                         str(resized_height)])
                    line_count += 1

        # Save data to new file
        file = open(file_path_write, 'w')
        with file:
            writer = csv.writer(file)
            writer.writerows(new_data)


    except:
        print("Error Occured")


# resize_data(train_file, train_file_new,train_img)
resize_data(test_file, os.getcwd() + "/data/polyps_rcnn/val_new.csv", test_img)
