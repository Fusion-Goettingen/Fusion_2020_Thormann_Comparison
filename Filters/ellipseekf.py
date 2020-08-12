import numpy as np

from matplotlib.patches import Ellipse

from Filters.basefilters import ExtendedObjectFilter
from Filters.filtersupport import get_proc_cov
from constants import *


class EllipseEKF(ExtendedObjectFilter):
    """
    Single target ellipse tracker based on RHM and polar ellipse equation. Can handle combined orientation for velocity
    and shape.

    Attributes
    ----------
    state       Current state, consisting of center, velocity, orientation, and semi-axes
    cov         Current state covariance
    sigma_q     Standard deviation of kinematic process noise
    sigma_sh    Standard deviation of shape process noise
    mode        Either normal or implicit
    pred_mode   Either coupled for polar velocity and single orientation or normal for Cartesian velocity
    al_approx   If true, orientation is not tracked and instead approximated by the Cartesian velocity
    x1          Index of position first dimension in state
    x2          Index of position second dimension in state
    v1          Index of velocity first dimension in state
    v2          Index of velocity second dimension in state
    v           Index of polar velocity in state if used
    al          Index of orientation in state
    l           Index of semi-axis length in state
    w           Index of semi-axis width in state
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._state = kwargs.get('init_state').copy()
        self._cov = kwargs.get('init_cov').copy()

        self._sigma_q = kwargs.get('sigma_q').copy()
        self._sigma_sh = kwargs.get('sigma_sh').copy()

        self._mode = kwargs.get('mode')
        self._pred_mode = kwargs.get('pred_mode')
        self._al_approx = kwargs.get('al_approx')  # approximate shape orientation via velocity vector
        self._meas_asso = kwargs.get('meas_asso')

        self._x1 = 0
        self._x2 = 1
        self._v1 = 2
        self._v2 = 3
        self._v = 2
        if self._pred_mode == 'coupled':
            if self._al_approx:
                print('Invalid mode combination regarding orientation. Removing orientation approximation.')
                self._al_approx = False
            self._al = 3
            self._l = 4
            self._w = 5
            self.couple_state()
            self._sigma_q = self._sigma_q[0]
        else:
            self._al = 4
            self._l = 5
            self._w = 6

    def reset(self, init_state, init_cov):
        self._est = init_state.copy()
        self._state = init_state.copy()
        self._cov = init_cov.copy()

        if self._pred_mode == 'coupled':
            self.couple_state()

    def couple_state(self):
        """
        Replace Cartesian velocity by polar velocity using shape orientation in state mean and covariance
        :return:
        """
        self._state = np.zeros(6)
        self._state[[self._x1, self._x2, self._al, self._l, self._w]] = self._est[[X1, X2, AL, L, W]]
        self._state[self._v] = np.linalg.norm(self._est[[V1, V2]])
        self._cov = self._cov[[X1, X2, V1, AL, L, W]][:, [X1, X2, V1, AL, L, W]]

    def predict(self, td):
        """
        Switch between the two prediction modes.
        :param td:  Time difference
        """
        if self._pred_mode == 'normal':
            self.predict_normal(td)
        elif self._pred_mode == 'coupled':
            self.predict_coupled(td)
        else:
            print('Invalid prediction mode')

    def predict_normal(self, td):
        """
        Predict kinematic state according to NCV model and add noise to shape covariance.
        :param td:  Time difference
        """
        # get noise covariance using current orientation
        proc_cov = get_proc_cov(self._sigma_q, self._sigma_sh, td)

        proc_mat = np.array([
            [1.0, 0.0, td, 0.0],
            [0.0, 1.0, 0.0, td],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._state[:4] = np.dot(proc_mat, self._state[:4])
        self._cov[:4, :4] = np.dot(np.dot(proc_mat, self._cov[:4, :4]), proc_mat.T)
        self._cov += proc_cov

        self._cov += self._cov.T
        self._cov *= 0.5

    def predict_coupled(self, td):
        """
        Prediction assuming polar velocity with orientation same as shape orientation. Use EKF to deal with
        non-linearities.
        :param td:  Time difference
        """
        alpha = self._al
        # get noise covariance using current orientation
        error_mat = np.array([(td ** 2 / 2) * np.cos(self._state[alpha]), (td ** 2 / 2)
                              * np.sin(self._state[alpha]), td, 0, 0, 0])
        proc_cov = np.outer(error_mat, error_mat) * (self._sigma_q ** 2)
        proc_cov[3:, 3:] = np.diag(self._sigma_sh) ** 2 * td **2
        proc_cov[:2, :2] += np.outer(np.array([(td ** 2 / 2) * np.cos(self._state[alpha] + 0.5 * np.pi), (td ** 2 / 2)
                                               * np.sin(self._state[alpha] + 0.5 * np.pi)]),
                                     np.array([(td ** 2 / 2) * np.cos(self._state[alpha] + 0.5 * np.pi), (td ** 2 / 2)
                                               * np.sin(self._state[alpha] + 0.5 * np.pi)]))

        self._state[[self._x1, self._x2]] = self._state[[self._x1, self._x2]] + self._state[self._v] * td \
                                            * np.array([np.cos(self._state[alpha]), np.sin(self._state[alpha])])

        jac_proc = np.array([
            [1.0, 0.0, np.cos(self._state[alpha]) * td, -np.sin(self._state[alpha]) * self._state[self._v] * td, 0.0,
             0.0],
            [0.0, 1.0, np.sin(self._state[alpha]) * td, np.cos(self._state[alpha]) * self._state[self._v] * td, 0.0,
             0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        self._cov = np.dot(np.dot(jac_proc, self._cov), jac_proc.T)
        self._cov += proc_cov

        self._cov += self._cov.T
        self._cov *= 0.5

    def meas_equation(self, y, s, greedy_center=None):
        """
        Measurement equation for ellipse RHM.
        :param y:   Current measurement
        :param s:   Scaling factor for surface distribution
        :param greedy_center:   If set, will use m as center for greedy angle association
        :return:    The global assumed measurement source, the measurement angle to the center, and the result of the
                    radial function
        """
        # vector from target center to measurement
        center = greedy_center if greedy_center is not None else self._state[[self._x1, self._x2]]
        yhat_vec = y - center

        # angle of expected measurement source (greedy)
        ang = (np.arctan2(yhat_vec[self._x2], yhat_vec[self._x1]) + np.pi) % (2 * np.pi) - np.pi
        ell_ang = ang - self._state[self._al]  # angle in local coordinates

        # radial function
        l = (self._state[self._l] * self._state[self._w]) / np.sqrt(((self._state[self._w] * np.cos(ell_ang)) ** 2) +
                                                                    ((self._state[self._l] * np.sin(ell_ang)) ** 2))
        # vector from center to expected surface point, given scaling factor s and global angle ang
        yhat = s * l * np.array([np.cos(ang), np.sin(ang)])

        return yhat + self._state[[self._x1, self._x2]], ang, l

    def correct(self, meas, meas_cov):
        """
        Switch between different correction modes and prepare estimate depending on for of state vector.
        :param meas:        The measurement batch
        :param meas_cov:    Measurement covariance
        """
        if self._mode == 'normal':
            self.correct_normal_ekf(meas, meas_cov)
        elif self._mode == 'imp':
            self.correct_imp_ekf(meas, meas_cov)
        else:
            print('Invalid mode')

        if self._pred_mode == 'normal':
            self._est = self._state.copy()
        elif self._pred_mode == 'coupled':
            self._est = np.zeros(7)
            self._est[[0, 1, 2, 4, 5, 6]] = self._state
            self._est[[self._v1, self._v2]] = self._state[self._v] * np.array([np.cos(self._state[self._al]),
                                                                               np.sin(self._state[self._al])])
        else:
            print('Invalid prediction mode')

    def correct_normal_ekf(self, meas, meas_cov):
        """
        Correction using explicit measurement model.
        :param meas:        The batch of measurements, processed sequentially
        :param meas_cov:    Measurement covariance
        :return:
        """
        nz = len(meas)  # number of measurements

        # go through measurements
        meas_mean = np.mean(meas, axis=0)
        for i in range(0, nz):
            if self._al_approx & (not (self._pred_mode == 'coupled')):
                self._state[self._al] = np.arctan2(self._state[self._v2], self._state[self._v1])

            greedy_center = meas_mean if self._meas_asso else None
            yhat, ang, l = self.meas_equation(meas[i], MU_S, greedy_center=greedy_center)
            innov = meas[i] - yhat  # innovation

            # calculate Jacobian========================================================================================
            jac = np.zeros((2, len(self._state)))

            # dh/dxc
            alpha_xc = np.array([meas[i, self._x2] - self._state[self._x2],
                                 -(meas[i, self._x1] - self._state[self._x1])]) \
                       / ((meas[i, self._x1] - self._state[self._x1]) ** 2
                          + (meas[i, self._x2] - self._state[self._x2]) ** 2)
            # r derived by alpha (-r_u is derived by psi)
            r_u = (self._state[self._l] * self._state[self._w] ** 3 - self._state[self._w] * self._state[self._l] ** 3) \
                  * 0.5 * np.sin(2 * (ang - self._state[self._al])) \
                  / np.sqrt(((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2)
                            + ((self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2)) ** 3
            r_xc = r_u * alpha_xc
            jac[:, [self._x1, self._x2]] = np.identity(2) + MU_S \
                                           * (np.einsum('a, b -> ab', np.array([np.cos(ang), np.sin(ang)]), r_xc)
                                              + (l * np.einsum('a, b -> ab', np.array([-np.sin(ang), np.cos(ang)]),
                                                               alpha_xc)))

            # dh/dpsi
            if not (self._al_approx & (not (self._pred_mode == 'coupled'))):
                jac[:, self._al] = -r_u * MU_S * np.array([np.cos(ang), np.sin(ang)])

            # dh/da and # dh/db
            r_a_part = self._state[self._w] / np.sqrt((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                                      + (self._state[self._l] * np.sin(
                ang - self._state[self._al])) ** 2)
            r_b_part = self._state[self._l] / np.sqrt((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                                      + (self._state[self._l] * np.sin(
                ang - self._state[self._al])) ** 2)
            jac[:, self._l] = MU_S * np.array([np.cos(ang), np.sin(ang)]) \
                              * (r_a_part - (self._state[self._l] ** 2 * self._state[self._w]
                                             * np.sin(ang - self._state[self._al]) ** 2)
                                 / (np.sqrt((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                            + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 3))
            jac[:, self._w] = MU_S * np.array([np.cos(ang), np.sin(ang)]) \
                              * (r_b_part - (self._state[self._w] ** 2 * self._state[self._l]
                                             * np.cos(ang - self._state[self._al]) ** 2)
                                 / (np.sqrt((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                                            + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 3))

            # Kalman update=============================================================================================
            yhat_local = yhat - self._state[[self._x1, self._x2]]
            innov_cov = np.einsum('ab, bc, dc -> ad', jac, self._cov, jac) + meas_cov \
                        + SIGMA_S * np.einsum('a, b -> ab', yhat_local, yhat_local)
            gain = np.einsum('ab, cb, cd -> ad', self._cov, jac, np.linalg.inv(innov_cov))

            self._state = self._state + np.einsum('ab, b -> a', gain, innov)
            self._cov = self._cov - np.einsum('ab, bc, dc -> ad', gain, innov_cov, gain)
            # if self._pred_mode != 'coupled':
            #     self._cov += np.identity(len(self._cov))*COV_ADD_EKF

            self._cov = (self._cov + self._cov.T) * 0.5  # avoid numerical problems

            self._state[self._al] = (self._state[self._al] + np.pi) % (2 * np.pi) - np.pi
            # lower threshold for shape parameters
            self._state[self._l] = np.max([AX_MIN, self._state[self._l]])
            self._state[self._w] = np.max([AX_MIN, self._state[self._w]])

    def correct_imp_ekf(self, meas, meas_cov):
        """
        Kalman filter correction step using the implicit measurement equation. If velocity orientation is separate from
        ellipse shape orientation, its derivatives in the Jacobian would be 0
        :param meas:        The measurement batch, processed sequentially
        :param meas_cov:    Measurement covariance
        """
        nz = len(meas)

        # go through measurements
        for i in range(0, nz):
            if self._al_approx & (not (self._pred_mode == 'coupled')):
                self._state[self._al] = np.arctan2(self._state[self._v2], self._state[self._v1])

            yhat, ang, l = self.meas_equation(meas[i], MU_S)

            # calculate Jacobians=======================================================================================
            mu_h = MU_S2 * l ** 2 - np.linalg.norm(meas[i] - self._state[[self._x1, self._x2]]) ** 2 + np.trace(
                meas_cov)

            jac_1 = np.zeros(len(self._state))
            dh_dalph = MU_S2 * (self._state[self._l] ** 2 * self._state[self._w] ** 4
                                - self._state[self._l] ** 4 * self._state[self._w] ** 2) \
                       * np.sin(2 * (ang - self._state[self._al])) \
                       / ((self._state[self._w] * np.cos(ang - self._state[self._al])) ** 2
                          + (self._state[self._l] * np.sin(ang - self._state[self._al])) ** 2) ** 2
            dalph_dxc = np.array([meas[i, self._x2] - self._state[self._x2],
                                  -(meas[i, self._x1] - self._state[self._x1])]) \
                        / ((meas[i, self._x1] - self._state[self._x1]) ** 2
                           + (meas[i, self._x2] - self._state[self._x2]) ** 2)
            jac_1[[self._x1, self._x2]] = dh_dalph * dalph_dxc
            if not self._al_approx & (not (self._pred_mode == 'coupled')):
                jac_1[self._al] = -dh_dalph

            a = self._state[self._l]
            b = self._state[self._w]
            psi = self._state[self._al]
            jac_1[self._l] = MU_S2 * (2 * a * b ** 2
                                      / ((b * np.cos(ang - psi)) ** 2
                                         + (a * np.sin(ang - psi)) ** 2)
                                      - 2 * a ** 3 * b ** 2
                                      * np.sin(ang - psi) ** 2
                                      / ((b * np.cos(ang - psi)) ** 2
                                         + (a * np.sin(ang - psi)) ** 2) ** 2)
            jac_1[self._w] = MU_S2 * (2 * b * a ** 2
                                      / ((b * np.cos(ang - psi)) ** 2
                                         + (a * np.sin(ang - psi)) ** 2)
                                      - 2 * b ** 3 * a ** 2
                                      * np.cos(ang - psi) ** 2
                                      / ((b * np.cos(ang - psi)) ** 2
                                         + (a * np.sin(ang - psi)) ** 2) ** 2)

            jac_2 = np.array([-2.0 * (meas[i, self._x1] - self._state[self._x1]),
                              -2.0 * (meas[i, self._x2] - self._state[self._x2]),
                              0.0, 0.0, 0.0, 0.0, 0.0])[:len(self._state)]

            t = 2.0 * (yhat - self._state[[self._x1, self._x2]])
            r_h3 = np.trace(np.dot(t, t) * meas_cov) + 2.0 * np.trace(np.dot(meas_cov, meas_cov)) \
                   + np.trace(meas_cov) ** 2
            r_h4 = SIGMA_S2 * l ** 4

            r_h = r_h3 + r_h4

            # Kalman update=============================================================================================
            cov_xh = np.dot(self._cov, (jac_1 - jac_2).T)
            cov_h = np.dot(np.dot(jac_1 - jac_2, self._cov), jac_1 - jac_2) + r_h
            self._state = self._state + cov_xh * -mu_h / cov_h
            self._cov = self._cov - np.einsum('a, b -> ab', cov_xh / cov_h, cov_xh)
            self._cov = (self._cov + self._cov.T) * 0.5

            self._state[self._al] = (self._state[self._al] + np.pi) % (2 * np.pi) - np.pi
            # lower threshold for shape parameters
            self._state[self._l] = np.max([AX_MIN, self._state[self._l]])
            self._state[self._w] = np.max([AX_MIN, self._state[self._w]])

    def plotting(self):
        """
        Plot current estimate.
        """
        ell = Ellipse((self._est[X1], self._est[X2]), self._est[L] * 2.0, self._est[W] * 2.0, np.rad2deg(self._est[AL]),
                      ec=self._color, fill=False, zorder=2)
        self._ax.add_artist(ell)
