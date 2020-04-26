"""
Reimplement evaluation.mat provided by Adobe in python
"""

import scipy.ndimage
import os
import cv2
import numpy as np
from skimage.measure import label
import scipy.ndimage.morphology
from glob import glob
from tqdm import tqdm


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, trimap):
    # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
    # target = (target - np.min(target)) / (np.max(target) - np.min(target) + 1e-8)

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    # error_map = (pred_x/pred_amp - target_x/target_amp) ** 2 + (pred_y/pred_amp - target_y/target_amp) ** 2
    loss = np.sum(error_map[trimap == 128])

    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap, step):
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    dist_maps = np.zeros([h, w, len(thresh_steps)], dtype=np.int)
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]
        # dist_maps[:,:,i] = scipy.ndimage.morphology.distance_transform_edt(omega)
        # dist_maps[:,:,i] = dist_maps[:,:,i] / (np.max(dist_maps[:,:,i])+1e-8)

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)
    loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])

    return loss / 1000.


def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)
    # loss = np.mean((error_map ** 2)[trimap == 128])

    return loss


def comput_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000


if __name__ == '__main__':
    # # pred_dir = 'vis_test/orig/alpha'
    # alpha_dir = '/versa/liyaoyi/dataset/AdobeMatting/test_set/mask'
    # trimap_dir = '/versa/liyaoyi/dataset/AdobeMatting/test_set/trimaps'

    # pred_dir = '/versa/liyaoyi/GPUGuidedFuilter/vis_test/alpha_trimap'
    # # pred_dir = '/versa/liyaoyi/dataset/AdobeMatting/test'
    # mask_dir = '/versa/liyaoyi/InductiveGuidedFilter/vis_test/orig/mask'
    # # alpha_dir = '/versa/liyaoyi/InductiveGuidedFilter/vis_test/orig/gt'
    ######

    alpha_dir = '/versa/liyaoyi/InductiveGuidedFilter/vis_test/orig/gt'
    trimap_dir = '/versa/liyaoyi/Deep-Image-Matting/vis_test/trimap_20'
    # pred_dir = '/versa/liyaoyi/Deep-Image-Matting/vis_test/alpha_worse'
    pred_dir = '/versa/liyaoyi/InductiveGuidedFilter/vis_test/orig/atten_gabor'
    # mask_dir = '/versa/liyaoyi/InductiveGuidedFilter/vis_test/orig/mask_worse'

    mse_loss = 0
    sad_loss = 0
    grad_loss = 0
    conn_loss = 0
    trimap_sum = 0
    num = 0
    for path in tqdm(glob(os.path.join(pred_dir, '*.png'))):
        num += 1
        name_part = os.path.basename(path)
        pred = cv2.imread(path, 0).astype(np.float)
        # mask = cv2.imread(os.path.join(mask_dir, name_part), 0).astype(np.float)
        trimap = cv2.imread(os.path.join(trimap_dir, name_part), 0)

        # alpha = cv2.imread(os.path.join(alpha_dir, '_'.join(name_part.split('_')[:-1])+'.png'), 0).astype(np.float)
        # trimap = np.ones_like(pred, dtype=np.uint8) * 128
        # mask_erode = cv2.erode((mask>128).astype(np.uint8), kernel=np.ones([20, 20]))
        # mask_dilate = cv2.dilate((mask>128).astype(np.uint8), kernel=np.ones([20, 20]))
        # trimap[mask_erode==1] = 255
        # trimap[mask_dilate==0] = 0

        alpha = cv2.imread(os.path.join(alpha_dir, name_part), 0)

        mse_loss += compute_mse_loss(pred, alpha, trimap)
        tmp = comput_sad_loss(pred, alpha, trimap)
        conn_loss += compute_connectivity_error(pred, alpha, trimap, 0.1)
        sad_loss += tmp[0]
        trimap_sum += tmp[1]
        grad_loss += compute_gradient_loss(pred, alpha, trimap)

    print('MSE: \t\t', mse_loss / num)
    print('SAD: \t\t', sad_loss / num)
    print('GRAD: \t\t', grad_loss / num)
    print('CONN: \t\t', conn_loss / num)
    print('TRIMAP SUM: \t\t', trimap_sum / num)
