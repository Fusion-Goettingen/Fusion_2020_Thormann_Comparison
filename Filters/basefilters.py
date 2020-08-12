import numpy as np

from ErrorAndPlotting.error import gw_error
from constants import *


class ExtendedObjectFilter:
    """
    Basic structure for the extended object tracking filters.

    Attributes
    ----------
    est         current estimate, must be [x1, x2, v1, v2, alpha, l, w]
    error_gw    Gaussian Wasserstein distance error for each time step
    error_vel   Velocity error for each time step
    name        The name of the tracker
    color       Color in which to plot the trackers estimates
    ax          Axis on which to plot
    """

    def __init__(self, **kwargs):
        # estimate of the filter, consisting of 2D center and velocity, orientation, and semi-axes
        self._est = kwargs.get('init_state').copy()
        # mean square Gaussian Wasserstein error for each time step
        self._error_gw = np.zeros(kwargs.get('time_steps'))
        # mean square error of velocity vector
        self._error_vel = np.zeros(kwargs.get('time_steps'))

        # for plotting
        self._name = kwargs.get('name')
        self._color = kwargs.get('color')
        self._ax = kwargs.get('ax')

    def get_name(self):
        return self._name

    def get_color(self):
        return self._color

    def step(self, meas, meas_cov, td, step_id, gt, plotting):
        if td > 0:
            self.predict(td)
        self.correct(meas, meas_cov)
        if plotting:
            self.plotting()

        self._error_gw[step_id] += gw_error(self._est, gt)
        self._error_vel[step_id] += np.dot(self._est[[V1, V2]] - gt[[V1, V2]], self._est[[V1, V2]] - gt[[V1, V2]])

    def predict(self, td):
        pass

    def correct(self, meas, meas_cov):
        pass

    def plotting(self):
        pass

    def reset(self, init_state, init_cov):
        pass

    def plot_gw_error(self, ax, runs):
        self._error_gw /= runs
        print("%s: %.4f" %(self._name, float(np.mean(self._error_gw))))
        ax.plot(np.arange(TIME_STEPS), self._error_gw, color=self._color, label=self._name)

    def plot_vel_error(self, ax, runs):
        self._error_vel /= runs
        ax.plot(np.arange(TIME_STEPS), self._error_vel, color=self._color, label=self._name)
