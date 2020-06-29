import numpy as np


# for simulation
LOAD_DATA = False
INIT_STATE = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 3.0, 1.5])
MEAS_COV = np.diag([0.5, 0.5])**2

# sharp turn
VELS = [10.0, 10.0, 10.0, 10.0, 10.0,  # first straight 0-5
        10.0, 10.0, 10.0, 10.0,  # light turn 6-9
        10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 7.0, 7.0, 7.0,  # straight with braking 10-18
        7.0, 7.0, 7.0,  # strong turn 19-21
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]  # final straight 22-32
ORS = [0.0, 0.0, 0.0, 0.0, 0.0,
       0.05*np.pi, 0.1*np.pi, 0.15*np.pi, 0.2*np.pi,
       0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi, 0.2*np.pi,
       0.3*np.pi, 0.4*np.pi, 0.5*np.pi,
       0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.5*np.pi,
       0.5*np.pi]

# simple trajectory (comment to use sharp turn)
VELS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
ORS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.01*np.pi, 0.02*np.pi, 0.03*np.pi, 0.04*np.pi, 0.05*np.pi,
       0.05*np.pi, 0.05*np.pi, 0.05*np.pi]

TIME_STEPS = len(VELS)+1  # number of time steps
TD = 1.0  # time difference between steps
GT_POIS = 40  # poisson rate of measurements generated by ground truth
RUNS = 1000

# for plotting
AX_LIMS = [-10, 160, -10, 160]

# for tracking
INIT_STATE_COV = np.diag([0.5, 0.5, 0.1, 0.1, 0.05*np.pi, 0.5, 0.2])**2
AX_MIN = 0.1  # minimum semi-axis length

# MEM-EKF*
MEM_H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])
MEM_KIN_DYM = np.array([
    [1.0, 0.0, TD,  0.0],
    [0.0, 1.0, 0.0,  TD],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
MEM_SHAPE_DYM = np.identity(3)

# Gaussian approximation of uniform scaling factor s
MU_S = 2.0 / 3.0  # expectation of s
SIGMA_S = 1.0 / 18.0  # standard deviation of s
MU_S2 = 1.0 / 2.0  # expectation of s^2
SIGMA_S2 = 1.0 / 12.0  # standard deviation of s^2

# indices for ground truth and estimate
X1 = 0
X2 = 1
V1 = 2
V2 = 3
AL = 4
L = 5
W = 6
