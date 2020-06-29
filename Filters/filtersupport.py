import numpy as np

from constants import *


def get_proc_cov(sigma_q, sigma_sh, td):
    """
    Calculate the process covariance based on kinematic and shape process noise.
    :param sigma_q:     standard deviation of kinematic process noise
    :param sigma_sh:    standard deviation of shape process noise
    :param td:          time difference between time steps
    :return:            the 7x7 process noise covariance
    """
    proc_cov = np.zeros((7, 7))

    error_mat = np.array([
        [0.5*td**2, 0.0],
        [0.0, 0.5*td**2],
        [td,        0.0],
        [0.0,        td],
    ])
    proc_cov[:4, :4] = np.dot(np.dot(error_mat, np.diag(sigma_q)**2), error_mat.T)

    proc_cov[4:, 4:] = td**2*np.diag(sigma_sh)**2

    return proc_cov
