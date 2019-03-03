import os
import argparse
import pprint
from PIL import Image
from object_detection.logging.logger import rootLogger
from object_detection.utils import (get_available_datasets)
from scipy import ndimage as ndi
import csv
import cv2
import scipy.misc
import numpy as np



extensions = ('.avi')


def extractframes(pathOut):

    """loop through directory and sub-directories to get all videos."""
    rootdir = os.path.join(os.getcwd(), 'data/' + args.dataset + '/GT_Samples/')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                print(os.path.join(subdir, file))

                """create folder for each video frames"""

                if not os.path.exists(pathOut+str(file)):
                    os.mkdir(pathOut+str(file))
                cap = cv2.VideoCapture(subdir+"/"+file)
                cap.get(cv2.CAP_PROP_FPS)
                count = 0

                while (cap.isOpened()):

                    """Capture frame-by-frame"""
                    ret, frame = cap.read()

                    if ret == True:
                        print('Read %d frame: ' % count, ret)
                        """save frame as a jpg file"""
                        cv2.imwrite(os.path.join(pathOut+str(file).format(), "frame{:d}.jpg".format(count)), frame)
                        count += 1
                    else:
                        break

                # When everything done, release the capture
                cap.release()
                cv2.destroyAllWindows()


def draw_boxes(img, bboxes, color=(0, 255, 0), thick=1):
    # Make a copy of the image
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        # print(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return img

# Datasets
DATASETS = {'polyps'}


# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Bounding Box Predictions')


# general
parser.add_argument('-d', '--dataset', type=str, default='polyps',
                    help="dataset, {'" +\
                         "', '".join(get_available_datasets()) +\
                         "'}")
parser.add_argument('--data-dirpath', type=str, default='data/',
                    help='directory for storing downloaded data')

# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())


def main(args=args):
    """

    main to extract images and bounding box for these images from csv files.

    comment the extractframes() function if the images are already extracted.

    """
    # pylint: disable=line-too-long
    csvlist = []
    final_box = list()
    FILE_PATH = os.path.join(os.getcwd(), 'data/' + args.dataset + '/GT_Samples/')
    images_out_path = os.path.join(os.getcwd(), 'data/' + args.dataset + '/GT_Samples/images/')
    bbox_images = os.path.join(os.getcwd(), 'data/' + args.dataset + '/GT_Samples/imageswithboxes/')

    "comment the line below if you already extracted the images"

    # extractframes(images_out_path)

    for filecsv in sorted(os.listdir(FILE_PATH)):
        if filecsv.endswith("csv"):
            csvplit = os.path.splitext(filecsv[0:-3])
            finalcsv = csvplit[0]
            with open(FILE_PATH + filecsv, mode='r') as bbox_file:
                bbox_reader = csv.reader(bbox_file, delimiter=',')
                # print(filecsv)
                for every in bbox_reader:
                    csvlist.append(every)
                # print(csvlist)
                for dir in sorted(os.listdir(images_out_path)):
                    # print(dir)
                    dirsplit = os.path.splitext(dir[0:-9])
                    finaldir = dirsplit[0]
                    if finalcsv == finaldir:
                        print(finalcsv, finaldir)
                        for image, each in zip(sorted(os.listdir(images_out_path+'/'+dir)), csvlist[1:]):

                            print(image)
                            print(each)
                            img = Image.open(images_out_path+'/'+dir+'/'+image)
                            imgarr = np.array(img)
                            final_box.append([(int(each[0]), int(each[2])), (int(each[1]), int(each[3]))])
                            labelled = draw_boxes(imgarr, final_box)
                            scipy.misc.imsave((bbox_images + dir + image), labelled)
                            final_box = list()
                csvlist = list()



if __name__ == '__main__':
    main()