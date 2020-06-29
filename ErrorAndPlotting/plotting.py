import numpy as np

from matplotlib.patches import Ellipse

from constants import *


def plot_ellipse(gt, meas, ax):
    """
    Plot an ellipse along with measurements.
    :param gt:      The ellipse to be plotted
    :param meas:    Measurement points
    :param ax:      Axis on which to plot
    """
    ell = Ellipse((gt[0], gt[1]), gt[5]*2.0, gt[6]*2.0, np.rad2deg(gt[4]), color='grey', zorder=1)
    ax.add_artist(ell)

    ax.scatter(meas[:, 0], meas[:, 1], s=0.1, color='black', zorder=3)
