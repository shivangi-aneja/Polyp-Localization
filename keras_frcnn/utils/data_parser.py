import csv
import os
import numpy as np
import cv2

from keras_frcnn.utils.logger import rootLogger


def get_data(input_path):
    classes_count = {}
    class_mapping = {}
    visualise = False

    # Get csv file for data
    train_annot_path = os.path.join(input_path, 'train.csv')
    val_annot_path = os.path.join(input_path, 'val.csv')

    # Get Image Path
    imgsets_path_train = os.path.join(input_path, 'train')
    imgsets_path_val = os.path.join(input_path, 'val')

    rootLogger.info('Parsing annotation files')

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
                elif np.abs(int(row[4])-int(row[6]))  > 50 and np.abs(int(row[5])-int(row[7]))  > 50:
                    print()
                    train_files.append(row[0])
                    annotation_data = {'filepath': os.path.join(imgsets_path_train, row[0]), 'width': row[1],
                                       'height': row[2],
                                       'bboxes': [
                                           {'class': row[3], 'x1': int(row[4]), 'x2': int(row[6]), 'y1': int(row[5]),
                                            'y2': int(row[7])}]}
                    train_annotation_data.append(annotation_data)
                    for data in annotation_data['bboxes']:
                        class_name = data['class']
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1

                        if class_name not in class_mapping:
                            class_mapping[class_name] = len(class_mapping)
                    if visualise:
                        img = cv2.imread(annotation_data['filepath'])
                        for bbox in annotation_data['bboxes']:
                            cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                        cv2.imshow('img', img)
                        cv2.waitKey(0)
    except Exception as e:
        rootLogger.info("Exception in loading training data")

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
                                       'height': row[2],
                                       'bboxes': [
                                           {'class': row[3], 'x1': int(row[4]), 'x2': int(row[6]), 'y1': int(row[5]),
                                            'y2': int(row[7])}]}
                    val_annotation_data.append(annotation_data)
                    for data in annotation_data['bboxes']:
                        class_name = data['class']
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1

                        if class_name not in class_mapping:
                            class_mapping[class_name] = len(class_mapping)
                    if visualise:
                        img = cv2.imread(annotation_data['filepath'])
                        for bbox in annotation_data['bboxes']:
                            cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                        cv2.imshow('img', img)
                        cv2.waitKey(0)
    except Exception as e:
        rootLogger.info("Exception in loading Validation data")

    # all_imgs = train_annotation_data + val_annotation_data
    return train_annotation_data, val_annotation_data, classes_count, class_mapping
