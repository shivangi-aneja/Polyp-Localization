# Calculate TP, FP , TN, FN and Precision Recall etc for segmentation mask

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from scipy.ndimage import label as sklabel
from skimage.util._montage import montage2d as montage


tp_threshold = 0.7
fp_threshold = 0.2

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def get_eval_metrics(label_trues, label_preds):

    # Find the region proposals

    tp_count = 0
    fp_count = 0
    fn_count = 0

    for i in range(len(label_trues)):
        hist_mat = _fast_hist(label_trues[i],label_preds[i],2)

        # Calculate True Positives and False Negative otherwise
        tp = hist_mat[1][1] / (hist_mat[1][1] + hist_mat[1][0])
        if tp >= tp_threshold:
            tp_count += 1
        else:
            fn_count += 1


        # Calculate FP (Can be multiple based on how many bounding boxes we have)

        # Get region proposals
        #props = regionprops(label_preds[i])

        label, num_features = sklabel(label_preds[i])

        for prop in range(num_features):
            pred_mask = np.array([], dtype='int64')
            gt_mask = np.array([],dtype='int64')
            mask_coords = np.argwhere(label==(prop+1))

            for mask in mask_coords:
                pred_mask = np.append(pred_mask,int(label_preds[i][mask[0]][mask[1]]))
                gt_mask = np.append(gt_mask,int(label_trues[i][mask[0]][mask[1]]))

            hist_mat_mask = _fast_hist(gt_mask, pred_mask, 2)
            fp = hist_mat_mask[0][1]/ (hist_mat_mask[0][1] + hist_mat_mask[1][1])
            if fp >= fp_threshold:
                fp_count += 1

        # print(tp_count)
        # print(fp_count)
        # print(fn_count)

    prec = tp_count / (tp_count + fp_count)
    rec = tp_count / (tp_count + fn_count)
    f1_score = (2 * prec * rec) / (prec + rec)
    return prec, rec, f1_score

        #Evaluate each proposal
        # for prop in props:
        #     pred_mask = np.array([],dtype='int64')
        #     gt_mask = np.array([],dtype='int64')
        #     mask_coords = prop['coords']
        #
        #     # Get the image pixels for the region proposal generated
        #     # row_index = mask_coords[:,0]
        #     # col_index = mask_coords[:,1]
        #     for mask in mask_coords:
        #         pred_mask = np.append(pred_mask,int(label_preds[i][mask[0]][mask[1]]))
        #         gt_mask = np.append(gt_mask,int(label_trues[i][mask[0]][mask[1]]))
        #
        #     hist_mat_mask = _fast_hist(gt_mask, pred_mask, 2)
        #     fp = hist_mat_mask[0][1]/ (hist_mat_mask[0][1] + hist_mat_mask[1][1])
        #     if fp >= fp_threshold:
        #         fp_count += 1
        #
        #     print(tp_count)
        #     print(fp_count)
        #     print(fn_count)