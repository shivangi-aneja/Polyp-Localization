#!/usr/bin/env python

import numpy as np
import csv
import os
import cv2

WARNING_AMOUNT_OF_BOXES=50

resize_dim=600

class Image:

    def __init__(self):
        pass


def get_data(input_path):
    classes_count = {}
    class_mapping = {}

    # Get csv file for data
    train_annot_path = os.path.join(input_path, 'train.csv')

    # Get Image Path
    imgsets_path_train = os.path.join(input_path, 'train')

    print('Parsing annotation files')

    train_files = []

    train_annotation_data = []
    #val_annotation_data = []

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
                    annotation_data =  ( (int(row[4]), int(row[5]), int(row[6]), int(row[7])) , row[1],row[2],row[0])
                    train_annotation_data.append(annotation_data)
                    # for data in annotation_data['bboxes']:
                    #     class_name = data['class']
                    #     if class_name not in classes_count:
                    #         classes_count[class_name] = 1
                    #     else:
                    #         classes_count[class_name] += 1
                    #
                    #     if class_name not in class_mapping:
                    #         class_mapping[class_name] = len(class_mapping)

    except Exception as e:
        print("Exception in loading training data")

    # all_imgs = train_annotation_data + val_annotation_data
    return train_annotation_data


def get_statistics():
    train_annot_path = os.getcwd()+ '/data/polyps_rcnn/'
    data = get_data(train_annot_path)
    images = []

    for boxes, width, height, filename in data:
        image=Image()
        image.filename=filename
        image.width=int(width)
        image.height=int(height)
        if width > height:
            image.dest_height = resize_dim
            image.scale_factor = resize_dim / float(image.height)
            image.dest_width = image.width * image.scale_factor
        else:
            image.dest_width = resize_dim
            image.scale_factor = resize_dim / float(image.width)
            image.dest_height = image.height * image.scale_factor

        image.boxes=[]
        image.resized_boxes=[]
        image.aspect_ratios=[]
        image.bbox_dims=[]

        #for val in boxes:
        x1, y1, x2, y2 = boxes
        image.boxes.append([x1,y1,x2,y2])

        x1_dest = int(x1 * image.scale_factor)
        y1_dest = int(y1 * image.scale_factor)
        x2_dest = int(x2 * image.scale_factor)
        y2_dest = int(y2 * image.scale_factor)
        image.resized_boxes.append([x1_dest,
                                    y1_dest,
                                    x2_dest,
                                    y2_dest
                                    ])
        image.bbox_dims.append((x2_dest-x1_dest, y2_dest-y1_dest))
        print(filename,x1,x2,y1,y2)
        image.aspect_ratios.append(float((x2 - x1) / float(y2 - y1)))

        images.append(image)
    return images


stats = get_statistics()

warnings=[]
for im in stats:
    print('image size: %s x %s  -> %s x %s'%(im.width,im.height, int(im.dest_width), int(im.dest_height)))
    if len(im.boxes) > WARNING_AMOUNT_OF_BOXES :
        warnings.append("Warning: more than %d boxes: %s"%(WARNING_AMOUNT_OF_BOXES, im.filename))
    elif len(im.boxes)==0:
        warnings.append("Warning: zero boxes: %s" % (im.filename))
    for box_idx in range(0,len(im.boxes)):
        x1, y1, x2, y2 = im.boxes[box_idx]
        dest_x1, dest_y1, dest_x2, dest_y2 = im.resized_boxes[box_idx]
        aspect_ratio=im.aspect_ratios[box_idx]
        width = (dest_x2-dest_x1)
        height = (dest_y2-dest_y1)

        if (width <= 0 or height <= 0):
            warnings.append("ERROR: height or width is 0 for "+im.filename)
        if (width*height < 10):
            warnings.append("Warning: contains small box: "+im.filename)
        print("bbox: ")
        print("    resize       : [%s,%s,%s,%s] -> [%s,%s,%s,%s]"%(x1,y1,x2,y2,dest_x1,dest_y1,dest_x2,dest_y2))
        print(" Polyp_width = %s Polyp_height = %s "%( (x2-x1), (y2-y1)  ))
        print("    width       : %d"%(dest_x2-dest_x1))
        print("    height       : %d" % (dest_y2 - dest_y1))
        print("    aspect ratio : %.2f"%aspect_ratio)

for msg in warnings:
    print('warning:',msg)

flatten = lambda l: [item for sublist in l for item in sublist]

dims= flatten([im.bbox_dims for im in stats])

aspect_ratios = flatten([im.aspect_ratios for im in stats])
bbox_widths = [w for w,h in dims ]
bbox_heights = [h for w,h in dims]
scale_factors = [im.scale_factor for im in stats]
widths=[im.width for im in stats]
heights=[im.height for im in stats]

print('min im width: %.2f '%(min(widths)))
print('mean im width: %.2f '%(np.mean(widths)))
print('max im width: %.2f '%(max(widths)))

print('min im height: %.2f '%(min(heights)))
print('mean im height: %.2f '%(np.mean(heights)))
print('max im height: %.2f '%(max(heights)))


print('min scale_factors: %.2f '%(min(scale_factors)))
print('mean scale_factors: %.2f '%(np.mean(scale_factors)))
print('max scale_factors: %.2f '%(max(scale_factors)))

print('min bbox width: %.2f '%(min(bbox_widths)))
print('mean bbox width: %.2f '%(np.mean(bbox_widths)))
print('max bbox width: %.2f '%(max(bbox_widths)))

print('min bbox height: %.2f '%(min(bbox_heights)))
print('mean bbox height: %.2f '%(np.mean(bbox_heights)))
print('max bbox height: %.2f '%(max(bbox_heights)))

print('min aspect ratio: %.2f '%(min(aspect_ratios)))
print('mean aspect ratio: %.2f '%(np.mean(aspect_ratios)))
print('max aspect ratio: %.2f '%(max(aspect_ratios)))