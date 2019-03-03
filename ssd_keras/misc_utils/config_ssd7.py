img_height = 512  # Height of the input images
img_width = 512  # Width of the input images
img_channels = 3  # Number of color channels of the input images
intensity_mean = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1  # Number of positive classes
scales = [0.08, 0.16, 0.32, 0.64,
          0.96]  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0]  # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True  # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0]  # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True  # Whether or not the model is supposed to use coordinates relative to the image size
l2_regularization = 0.0005
