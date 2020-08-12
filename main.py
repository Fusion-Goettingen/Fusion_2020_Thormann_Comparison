import matplotlib.pyplot as plt
import tikzplotlib
import os

from numpy.random import multivariate_normal as mvn

import time

from configs import get_configs
from Filters.ellipseekf import EllipseEKF
from Filters.memekfstar import MemEkfStarTracker
from Filters.independentaxisestimation import IndependentAxisEstimation
from Data.simulation import simulate_data
from ErrorAndPlotting.plotting import plot_ellipse
from constants import *

# setup
_, ax = plt.subplots(1, 1)
init_state = INIT_STATE
init_state[L] = np.max([init_state[L], AX_MIN])
init_state[W] = np.max([init_state[W], AX_MIN])

# setup folders
paths = ["simData", "plots"]
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)

# for reproducibility
if LOAD_DATA:
    init_state = np.load('./simData/initState0.npy')
else:
    np.save('./simData/initState0.npy', init_state)

# tracker setup
config_ellipseekf_normal, config_ellipseekf_imp, config_ellipseekf_normal_coupled, config_ellipseekf_imp_coupled,\
    config_ellipseekf_imp_oa, config_memekfstar, config_memekfstar_oa, config_halfaxis, config_fixed, \
    config_ellipseekf_fixed_c, config_ellipseekf_fixed_oa = get_configs(init_state, ax)

ellipseekf_normal = EllipseEKF(**config_ellipseekf_normal)
ellipseekf_imp = EllipseEKF(**config_ellipseekf_imp)
ellipseekf_fixed = EllipseEKF(**config_fixed)
ellipseekf_fixed_c = EllipseEKF(**config_ellipseekf_fixed_c)
ellipseekf_fixed_oa = EllipseEKF(**config_ellipseekf_fixed_oa)


ellipseekf_normal_coupled = EllipseEKF(**config_ellipseekf_normal_coupled)
ellipseekf_imp_coupled = EllipseEKF(**config_ellipseekf_imp_coupled)

ellipseekf_imp_oa = EllipseEKF(**config_ellipseekf_imp_oa)

memekfstar = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV, **config_memekfstar)
memekfstar_oa = MemEkfStarTracker(MEM_H, MEM_KIN_DYM, MEM_SHAPE_DYM, MEAS_COV, **config_memekfstar_oa)

half_axis = IndependentAxisEstimation(**config_halfaxis)

# timing (note that not all algorithms have been optimized)
ellipseekf_normal_time = 0.0
ellipseekf_fixed_time = 0.0
ellipseekf_fixed_c_time = 0.0
ellipseekf_fixed_oa_time = 0.0
ellipseekf_imp_time = 0.0
ellipseekf_normal_coupled_time = 0.0
ellipseekf_imp_coupled_time = 0.0
ellipseekf_imp_oa_time = 0.0
memekfstar_time = 0.0
memekfstar_oa_time = 0.0
half_axis_time = 0.0

for r in range(RUNS):
    print('Starting run %i of %i' % (r+1, RUNS))
    plot_cond = (r == RUNS-1)
    # generate data
    simulator = simulate_data(INIT_STATE, MEAS_COV)

    # run filters
    step_id = 0
    for gt, meas in simulator:
        # print('Starting time step %i' % step_id)
        if LOAD_DATA:
            gt = np.load('./simData/gt%i-%i.npy' % (step_id, r))
            meas = np.load('./simData/meas%i-%i.npy' % (step_id, r))
        if step_id == 0:
            td = 0
        else:
            td = TD
        tic = time.perf_counter()
        ellipseekf_normal.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_normal_time += toc-tic

        tic = time.perf_counter()
        ellipseekf_fixed.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_fixed_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_imp.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_imp_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_fixed_oa.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_fixed_oa_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_fixed_c.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_fixed_c_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_normal_coupled.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_normal_coupled_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_imp_coupled.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_imp_coupled_time += toc - tic

        tic = time.perf_counter()
        ellipseekf_imp_oa.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        ellipseekf_imp_oa_time += toc - tic

        tic = time.perf_counter()
        memekfstar.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        memekfstar_time += toc - tic

        tic = time.perf_counter()
        memekfstar_oa.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        memekfstar_oa_time += toc - tic

        tic = time.perf_counter()
        half_axis.step(meas.copy(), MEAS_COV.copy(), td, step_id, gt, plot_cond)
        toc = time.perf_counter()
        half_axis_time += toc - tic

        if not LOAD_DATA:
            np.save('./simData/meas%i-%i.npy' % (step_id, r), meas)
            np.save('./simData/gt%i-%i.npy' % (step_id, r), gt)

        if plot_cond:
            plot_ellipse(gt, meas, ax)
        step_id += 1
    if LOAD_DATA:
        init_state = np.load('./simData/initState%i.npy' % (r+1))
    else:
        init_state = mvn(INIT_STATE, INIT_STATE_COV)
        np.save('./simData/initState%i.npy' % (r+1), init_state)
    ellipseekf_normal.reset(init_state, INIT_STATE_COV)
    ellipseekf_imp.reset(init_state, INIT_STATE_COV)
    ellipseekf_normal_coupled.reset(init_state, INIT_STATE_COV)
    ellipseekf_imp_coupled.reset(init_state, INIT_STATE_COV)
    ellipseekf_imp_oa.reset(init_state, INIT_STATE_COV)
    memekfstar.reset(init_state, INIT_STATE_COV)
    memekfstar_oa.reset(init_state, INIT_STATE_COV)
    half_axis.reset(init_state, INIT_STATE_COV)
    ellipseekf_fixed.reset(init_state, INIT_STATE_COV)
    ellipseekf_fixed_c.reset(init_state, INIT_STATE_COV)
    ellipseekf_fixed_oa.reset(init_state, INIT_STATE_COV)

# example trajectory plotting
plt.axis(AX_LIMS)
plt.plot([0], [0], color=ellipseekf_normal.get_color(), label=ellipseekf_normal.get_name())
plt.plot([0], [0], color=ellipseekf_imp.get_color(), label=ellipseekf_imp.get_name())
plt.plot([0], [0], color=ellipseekf_normal_coupled.get_color(), label=ellipseekf_normal_coupled.get_name())
plt.plot([0], [0], color=ellipseekf_imp_coupled.get_color(), label=ellipseekf_imp_coupled.get_name())
plt.plot([0], [0], color=ellipseekf_imp_oa.get_color(), label=ellipseekf_imp_oa.get_name())
plt.plot([0], [0], color=memekfstar.get_color(), label=memekfstar.get_name())
plt.plot([0], [0], color=memekfstar_oa.get_color(), label=memekfstar_oa.get_name())
plt.plot([0], [0], color=half_axis.get_color(), label=half_axis.get_name())
plt.plot([0], [0], color=ellipseekf_fixed.get_color(), label=ellipseekf_fixed.get_name())
plt.plot([0], [0], color=ellipseekf_fixed_c.get_color(), label=ellipseekf_fixed_c.get_name())
plt.plot([0], [0], color=ellipseekf_fixed_oa.get_color(), label=ellipseekf_fixed_oa.get_name())
plt.legend()
tikzplotlib.save('./plots/examplerun.tex', add_axis_environment=False)
plt.title("Example run")
plt.show()

# time wrap up
ellipseekf_normal_time /= TIME_STEPS*RUNS
ellipseekf_imp_time /= TIME_STEPS*RUNS
ellipseekf_normal_coupled_time /= TIME_STEPS*RUNS
ellipseekf_imp_coupled_time /= TIME_STEPS*RUNS
ellipseekf_imp_oa_time /= TIME_STEPS*RUNS
memekfstar_time /= TIME_STEPS*RUNS
memekfstar_oa_time /= TIME_STEPS*RUNS
half_axis_time /= TIME_STEPS*RUNS
ellipseekf_fixed_time /= TIME_STEPS*RUNS
ellipseekf_fixed_c_time /= TIME_STEPS*RUNS
ellipseekf_fixed_oa_time /= TIME_STEPS*RUNS

ticks = [ellipseekf_normal.get_name(), ellipseekf_imp.get_name(), ellipseekf_normal_coupled.get_name(),
         ellipseekf_imp_coupled.get_name(), ellipseekf_imp_oa.get_name(), memekfstar.get_name(),
         memekfstar_oa.get_name(), half_axis.get_name(), ellipseekf_fixed.get_name(), ellipseekf_fixed_c.get_name(),
         ellipseekf_fixed_oa.get_name()]
colors = [ellipseekf_normal.get_color(), ellipseekf_imp.get_color(), ellipseekf_normal_coupled.get_color(),
          ellipseekf_imp_coupled.get_color(), ellipseekf_imp_oa.get_color(), memekfstar.get_color(),
          memekfstar_oa.get_color(), half_axis.get_color(), ellipseekf_fixed.get_color(),
          ellipseekf_fixed_c.get_color(), ellipseekf_fixed_oa.get_color()]
runtimes = [ellipseekf_normal_time, ellipseekf_imp_time, ellipseekf_normal_coupled_time,
            ellipseekf_imp_coupled_time, ellipseekf_imp_oa_time, memekfstar_time,
            memekfstar_oa_time, half_axis_time, ellipseekf_fixed_time, ellipseekf_fixed_c_time,
            ellipseekf_fixed_oa_time]
bars = np.arange(1, len(ticks)+1, 1)
for i in range(len(ticks)):
    plt.bar(bars[i], runtimes[i], width=0.5, color=colors[i], label=ticks[i], align='center')
# plt.xticks(bars, ticks)
plt.legend()
tikzplotlib.save('./plots/runtimes.tex', add_axis_environment=False)
plt.savefig('./plots/runtimes.svg')
plt.title("Runtimes")
plt.show(block=False)

# error wrap up
_, ax = plt.subplots(1, 1)
ellipseekf_normal.plot_gw_error(ax, RUNS)
ellipseekf_imp.plot_gw_error(ax, RUNS)
ellipseekf_normal_coupled.plot_gw_error(ax, RUNS)
ellipseekf_imp_coupled.plot_gw_error(ax, RUNS)
ellipseekf_imp_oa.plot_gw_error(ax, RUNS)
memekfstar.plot_gw_error(ax, RUNS)
memekfstar_oa.plot_gw_error(ax, RUNS)
half_axis.plot_gw_error(ax, RUNS)
ellipseekf_fixed.plot_gw_error(ax, RUNS)
ellipseekf_fixed_c.plot_gw_error(ax, RUNS)
ellipseekf_fixed_oa.plot_gw_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/gw_error.tex', add_axis_environment=False)
plt.title("gw_error")
plt.show(block=False)

_, ax = plt.subplots(1, 1)
ellipseekf_normal.plot_vel_error(ax, RUNS)
ellipseekf_imp.plot_vel_error(ax, RUNS)
ellipseekf_normal_coupled.plot_vel_error(ax, RUNS)
ellipseekf_imp_coupled.plot_vel_error(ax, RUNS)
ellipseekf_imp_oa.plot_vel_error(ax, RUNS)
memekfstar.plot_vel_error(ax, RUNS)
memekfstar_oa.plot_vel_error(ax, RUNS)
half_axis.plot_vel_error(ax, RUNS)
ellipseekf_fixed.plot_vel_error(ax, RUNS)
ellipseekf_fixed_c.plot_vel_error(ax, RUNS)
ellipseekf_fixed_oa.plot_vel_error(ax, RUNS)
plt.legend()
tikzplotlib.save('./plots/vel_error.tex', add_axis_environment=False)
plt.title("vel_error")
plt.show()
