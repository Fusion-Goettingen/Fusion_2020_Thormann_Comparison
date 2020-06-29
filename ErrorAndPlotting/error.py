import numpy as np
from scipy.linalg import sqrtm

from Data.simulation import rot
from constants import *


def to_matrix(alpha, ax_l, ax_w, sr):
    """
    Turn ellipse parameters into a matrix or square root matrix depending on sr parameter
    :param alpha:   Orientation of the ellipse
    :param ax_l:    Semi-axis length of the ellipse
    :param ax_w:    Semi-axis width of the ellipse
    :param sr:      If True, square root matrix is calculated instead of shape matrix
    :return:        Shape or square root matrix depending of sr
    """
    p = 1 if sr else 2
    rot_m = rot(alpha)
    return np.dot(np.dot(rot_m, np.diag([ax_l, ax_w]) ** p), rot_m.T)


def gw_error(x, gt):
    """
    Calculates the squared Gaussian Wasserstein metric for two ellipses.
    :param x:   first ellipse, must be parameterized with center, orientation, and semi-axes
    :param gt:  second ellipse, must be parameterized with center, orientation, and semi-axes
    :return:    the Gaussian Wasserstein distance between the two ellipses
    """
    gt_sigma = to_matrix(gt[AL], gt[L], gt[W], False)
    gt_sigma += gt_sigma.T
    gt_sigma /= 2.0

    track_sigma = to_matrix(x[AL], x[L], x[W], False)
    track_sigma += track_sigma.T
    track_sigma /= 2.0

    error = np.linalg.norm(gt[[X1, X2]] - x[[X1, X2]]) ** 2 \
            + np.trace(gt_sigma + track_sigma - 2 * sqrtm(np.einsum('ab, bc, cd -> ad', sqrtm(gt_sigma), track_sigma,
                                                                    sqrtm(gt_sigma))))

    return error
