import csv
import os
from os import listdir
from os.path import isfile, join

import cv2

filepath = os.getcwd() + "/data/polyps_hospital/videos/"
file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
dir_name = "images"
img_width = 1920
img_height = 1080

try:
    if not os.path.exists(os.getcwd() + '/data/polyps_hospital/' + dir_name):
        os.makedirs(os.getcwd() + '/data/polyps_hospital/' + dir_name)
except OSError:
    print('Error: Creating directory of data')

rowList = []
for file in file_list:

    # Extract frames from the video
    nameList = []

    videoFile = cv2.VideoCapture(filepath + file)
    i = 1
    while videoFile.isOpened():
        ret, frame = videoFile.read()
        if ret is False:
            break
        name = os.getcwd() + '/data/polyps_hospital/' + dir_name + '/' + str(file.split('.')[0]) + "_" + str(
            int(i)) + '.png'
        nameList.append(str(file.split('.')[0]) + "_" + str(int(i)) + '.png')
        print("Creating " + name)
        cv2.imwrite(name, frame)
        i += 1
    videoFile.release()
    cv2.destroyAllWindows()

    # Read Coordinates for the video
    with open(os.getcwd() + "/data/polyps_hospital/csv/" + str(file.split('.')[0]) + ".csv") as bbox_reader:
        csv_reader = csv.reader(bbox_reader, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(row)
                new_row = [nameList[line_count - 1], img_width, img_height, 'polyp', row[0], row[2], row[1], row[3]]
                rowList.append(new_row)
                line_count += 1
        print(f'Processed {line_count} lines.')

# Write to csv file
header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
with open(os.getcwd() + "/data/polyps_hospital/bboxes.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header)  # write the header
    # write the actual content line by line
    for l in rowList:
        writer.writerow(l)
