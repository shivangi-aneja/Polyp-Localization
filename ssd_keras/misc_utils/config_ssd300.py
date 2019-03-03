img_height = 512  # Height of the model input images
img_width = 512  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images
mean_color = [0, 0,
              0]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1,
                 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_1 = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
            1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_2 = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87,
            1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_2
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
l2_regularization = 0.0005
confidence_threshold = 0.5
