import numpy as np

from matplotlib.patches import Ellipse

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import get_proc_cov
from constants import *


class IndependentAxisEstimation(ExtendedObjectFilter):
    """
    The Independent Axis Estimation filter considering sensor noise. Based on:
    Govaers, Felix. "On Independent Axes Estimation for Extended Target Tracking." 2019 Sensor Data Fusion: Trends,
    Solutions, Applications (SDF). IEEE, 2019.

    Attributes
    ----------
    x1          Position first dimension
    x2          Position second dimension
    v1          Velocity first dimension
    v2          Velocity second dimension
    l           Semi-axis length
    w           Semi-axis width
    h           Measurement matrix
    cov_kin     Kinematic state covariance
    cov_l       Variance of semi-axis length state
    cov_w       Variance of semi-axis width state
    sigma_q     Standard deviation of kinematic process noise
    sigma_sh    Standard deviation of shape process noise
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._x1 = 0
        self._x2 = 1
        self._v1 = 2
        self._v2 = 3

        self._state_kin = kwargs.get('init_state')[:4]
        self._l = kwargs.get('init_state')[L]
        self._w = kwargs.get('init_state')[W]

        self._h = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        self._cov_kin = kwargs.get('init_cov')[:4, :4]
        self._cov_l = kwargs.get('init_cov')[L, L]
        self._cov_w = kwargs.get('init_cov')[W, W]

        self._sigma_q = kwargs.get('sigma_q').copy()
        self._sigma_sh = kwargs.get('sigma_sh').copy()

    def reset(self, init_state, init_cov):
        self._state_kin = init_state[:4]
        self._l = init_state[L]
        self._w = init_state[W]

        self._cov_kin = init_cov[:4, :4]
        self._cov_l = init_cov[L, L]
        self._cov_w = init_cov[W, W]

    def predict(self, td):
        """
        Predict kinematic state based on NCV model and add process noise to shape.
        :param td:  Time difference
        """
        proc_cov = get_proc_cov(self._sigma_q, self._sigma_sh, td)

        proc_mat = np.array([
            [1.0, 0.0, td,  0.0],
            [0.0, 1.0, 0.0,  td],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._state_kin[:4] = np.dot(proc_mat, self._state_kin[:4])
        self._cov_kin[:4, :4] = np.dot(np.dot(proc_mat, self._cov_kin[:4, :4]), proc_mat.T)
        self._cov_kin += proc_cov[:4, :4]

        self._cov_kin += self._cov_kin.T
        self._cov_kin *= 0.5

        self._cov_l += td ** 2 * self._sigma_sh[1] ** 2
        self._cov_w += td ** 2 * self._sigma_sh[2] ** 2

    def correct(self, meas, meas_cov):
        """
        Correction step. Takes batch of measurements.
        :param meas:        Batch of measurements.
        :param meas_cov:    Measurement covariance
        """
        # update kinematic state
        alpha = np.arctan2(self._state_kin[self._v2], self._state_kin[self._v1])
        rot_mat = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha),  np.cos(alpha)],
        ])
        shape_mat = np.dot(np.dot(rot_mat, np.diag([self._l, self._w])), rot_mat.T)

        z_hat = np.mean(meas, axis=0)

        innov_cov = np.dot(np.dot(self._h, self._cov_kin), self._h.T) + (shape_mat * 0.25 + meas_cov) / len(meas)
        gain = np.dot(np.dot(self._cov_kin, self._h.T), np.linalg.inv(innov_cov))
        self._state_kin = self._state_kin + np.dot(gain, z_hat - self._state_kin[[self._x1, self._x2]])
        self._cov_kin = self._cov_kin - np.dot(np.dot(gain, innov_cov), gain.T)

        # update shape parameters only if enough measurements are there
        if len(meas) > 2:
            alpha = np.arctan2(self._state_kin[self._v2], self._state_kin[self._v1])
            rot_mat = np.array([
                [np.cos(alpha), -np.sin(alpha)],
                [np.sin(alpha), np.cos(alpha)],
            ])
            mu_tran = np.dot(np.dot(rot_mat.T, meas_cov), rot_mat) * 4.0
            z_mat = np.einsum('xa, xb -> ab', meas - z_hat[None, :], meas - z_hat[None, :]) / len(meas)

            # determine eigenvalues in most likely order
            eigs, vecs = np.linalg.eig(z_mat * 4.0)
            eig0_or_diff = np.minimum(abs(((np.arctan2(vecs[1, 0], vecs[0, 0]) - alpha) + np.pi) % (2*np.pi) - np.pi),
                                      abs(((np.arctan2(-vecs[1, 0], -vecs[0, 0]) - alpha) + np.pi) % (2*np.pi) - np.pi))
            eig1_or_diff = np.minimum(abs(((np.arctan2(vecs[1, 1], vecs[0, 1]) - alpha) + np.pi) % (2*np.pi) - np.pi),
                                      abs(((np.arctan2(-vecs[1, 1], -vecs[0, 1]) - alpha) + np.pi) % (2*np.pi) - np.pi))
            if eig0_or_diff > eig1_or_diff:  # switch eigenvalues to make R==V assumption possible
                eig_save = eigs[0]
                eigs[0] = eigs[1]
                eigs[1] = eig_save
            l_hat = np.sqrt(np.maximum(eigs[0] - mu_tran[0, 0], AX_MIN**2))
            w_hat = np.sqrt(np.maximum(eigs[1] - mu_tran[1, 1], AX_MIN**2))
            mu_l = (l_hat**2 + mu_tran[0, 0])**2 / (2*(len(meas) - 1)*l_hat**2)
            mu_w = (w_hat ** 2 + mu_tran[1, 1]) ** 2 / (2 * (len(meas) - 1) * w_hat ** 2)

            innov_cov_l = self._cov_l + mu_l
            gain_l = self._cov_l / innov_cov_l
            self._l = self._l + gain_l * (l_hat - self._l)
            self._cov_l = self._cov_l - gain_l**2 * innov_cov_l

            innov_cov_w = self._cov_w + mu_w
            gain_w = self._cov_w / innov_cov_w
            self._w = self._w + gain_w * (w_hat - self._w)
            self._cov_w = self._cov_w - gain_w ** 2 * innov_cov_w

        self._est = np.array([self._state_kin[self._x1], self._state_kin[self._x2], self._state_kin[self._v1],
                              self._state_kin[self._v2], alpha, self._l, self._w])

    def plotting(self):
        """
        Plot current estimate.
        """
        ell = Ellipse((self._est[X1], self._est[X2]), self._est[L] * 2.0, self._est[W] * 2.0, np.rad2deg(self._est[AL]),
                      ec=self._color, fill=False, zorder=2)
        self._ax.add_artist(ell)
