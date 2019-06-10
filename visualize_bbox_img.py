#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File for visualizing bbox in image
"""
import csv
import os

import cv2


def main():
    dataset = 'polyps_hospital'
    FILE_PATH = os.path.join(os.getcwd(), 'data/' + dataset + '/images/')
    FILE_PATH_NEW = os.path.join(os.getcwd(), 'data/' + dataset + '/bboxes/')
    csv_file = os.path.join(os.getcwd(), "data/" + dataset + "/bboxes.csv")

    with open(csv_file) as bbox_reader:
        csv_reader = csv.reader(bbox_reader, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(row)
                img = draw_boxes(img=cv2.imread(FILE_PATH + row[0]),
                                 bboxes=[(int(row[4]), int(row[5])), (int(row[6]), int(row[7]))])
                cv2.imwrite(filename=FILE_PATH_NEW + row[0], img=img)
                line_count += 1
        print(f'Processed {line_count} lines.')


def draw_boxes(img, bboxes, color=(0, 255, 0), thick=3):
    # Make a copy of the image
    # Iterate through the bounding boxes
    # for bbox in bboxes:
    # Draw a rectangle given bbox coordinates
    # print(bbox)
    cv2.rectangle(img, bboxes[0], bboxes[1], color, thick)
    # Return the image copy with boxes drawn
    return img


if __name__ == '__main__':
    main()
