# -*- coding: utf-8 -*-

"""
 File for training SSD 300 model
"""
from tensorflow import keras

K = keras.backend
Input = keras.layers.Input
Model = keras.models.Model
Progbar = keras.utils.Progbar
Adam = keras.optimizers.Adam
CSVLogger = keras.callbacks.CSVLogger
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
TerminateOnNaN = keras.callbacks.TerminateOnNaN
LearningRateScheduler = keras.callbacks.LearningRateScheduler
from math import ceil
import pprint

load_model = keras.models.load_model
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from models.keras_ssd512 import ssd_512
from misc_utils import config_ssd512 as Config
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Datasets
DATASETS = {'polyps_hospital'}

# training settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Create h5py files')

# general
parser.add_argument('-d', '--dataset', type=str, default='polyps_hospital',
                    help="dataset, {'" + \
                         "', '".join(sorted(DATASETS)) + \
                         "'}")
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('-e', '--final_epoch', type=int, default=1000,
                    help='Number Of Epochs')
parser.add_argument('-m', '--model_name', type=str, default='default',
                    help="model name to save")
parser.add_argument('-p', '--predict_mode', type=str, default='train',
                    help="prediction mode")
parser.add_argument('-tf', '--tf_logs', type=str, default='ssd512',
                    help="folder for tensorflow logging")
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('-gpu', '--gpu', type=int, default=0,
                    help="ID of the GPU to train on (or -1 to train on CPU)")

TF_LOG_PATH = os.path.join(os.getcwd(), 'tf_logs/')
# parse and validate parameters
args = parser.parse_args()

for k, v in args._get_kwargs():
    if isinstance(v, str):
        setattr(args, k, v.strip().lower())

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if args.gpu > -1 else '-1'

pprint.pprint(vars(args))


def main():
    create_new_model = True if args.model_name == 'default' else False

    if create_new_model:
        K.clear_session()  # Clear previous models from memory.
        model = ssd_512(image_size=(Config.img_height, Config.img_width, Config.img_channels),
                        n_classes=Config.n_classes,
                        mode='training', l2_regularization=Config.l2_regularization,
                        scales=Config.scales, aspect_ratios_per_layer=Config.aspect_ratios,
                        two_boxes_for_ar1=Config.two_boxes_for_ar1, steps=Config.steps, offsets=Config.offsets,
                        clip_boxes=Config.clip_boxes, variances=Config.variances,
                        normalize_coords=Config.normalize_coords,
                        subtract_mean=Config.mean_color, swap_channels=Config.swap_channels)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    else:

        model_path = "weights/" + args.model_name + ".h5"
        # We need to create an SSDLoss object in order to pass that to the model loader.
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        K.clear_session()  # Clear previous models from memory.

        model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                       'L2Normalization': L2Normalization,
                                                       'compute_loss': ssd_loss.compute_loss})

    # Load the data
    train_dataset = DataGenerator(load_images_into_memory=True,
                                  hdf5_dataset_path=os.getcwd() + "/data/" + args.dataset + '/polyp_train.h5')
    val_dataset = DataGenerator(load_images_into_memory=True,
                                hdf5_dataset_path=os.getcwd() + "/data/" + args.dataset + '/polyp_val.h5')
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size = val_dataset.get_dataset_size()
    print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

    batch_size = args.batch_size

    # For the training generator:
    ssd_data_augmentation = SSDDataAugmentation(img_height=Config.img_height, img_width=Config.img_width,
                                                background=Config.mean_color)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=Config.img_height, width=Config.img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=Config.img_height, img_width=Config.img_width,
                                        n_classes=Config.n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=Config.scales,
                                        aspect_ratios_per_layer=Config.aspect_ratios,
                                        two_boxes_for_ar1=Config.two_boxes_for_ar1,
                                        steps=Config.steps,
                                        offsets=Config.offsets,
                                        clip_boxes=Config.clip_boxes,
                                        variances=Config.variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=Config.normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
    train_generator = train_dataset.generate(batch_size=batch_size, shuffle=True,
                                             transformations=[ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images', 'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    model_checkpoint = ModelCheckpoint(
        filepath=os.getcwd() + '/weights/ssd512_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=30)

    csv_logger = CSVLogger(filename='ssd512_training_log.csv',
                           separator=',',
                           append=True)
    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
    terminate_on_nan = TerminateOnNaN()

    tf_log = keras.callbacks.TensorBoard(log_dir=TF_LOG_PATH + args.tf_logs, histogram_freq=0, batch_size=batch_size,
                                         write_graph=True, write_grads=False,
                                         write_images=False)

    callbacks = [model_checkpoint,
                 csv_logger,
                 learning_rate_scheduler,
                 terminate_on_nan, tf_log]

    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = 0
    final_epoch = args.final_epoch
    steps_per_epoch = 500

    # Train/Fit the model
    if args.predict_mode == 'train':
        history = model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=final_epoch,
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=ceil(val_dataset_size / batch_size), initial_epoch=initial_epoch)

    # Prediction Output
    predict_generator = val_dataset.generate(batch_size=1,
                                             shuffle=True,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'filenames',
                                                      'inverse_transform',
                                                      'original_images',
                                                      'original_labels'},
                                             keep_images_without_gt=False)

    i = 0
    for val in range(val_dataset_size):
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(
            predict_generator)

        y_pred = model.predict(batch_images)

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=Config.normalize_coords,
                                           img_height=Config.img_height,
                                           img_width=Config.img_width)

        # 5: Convert the predictions for the original image.
        y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded_inv[i])

        plt.figure(figsize=(20, 12))
        plt.imshow(batch_images[i])

        current_axis = plt.gca()

        colors = plt.cm.hsv(np.linspace(0, 1, Config.n_classes + 1)).tolist()  # Set the colors for the bounding boxes
        classes = ['background', 'polyps']  # Just so we can print class names onto the image instead of IDs

        for box in batch_original_labels[i]:
            xmin = box[1]
            ymin = box[2]
            xmax = box[3]
            ymax = box[4]
            label = '{}'.format(classes[int(box[0])])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white',
                              bbox={'facecolor': 'green', 'alpha': 1.0})

        for box in y_pred_decoded_inv[i]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        image = plt.gcf()
        plt.draw()
        image.savefig(os.getcwd() + "/val_ssd512val_" + str(val) + ".png", dpi=100)


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


if __name__ == '__main__':
    main()
