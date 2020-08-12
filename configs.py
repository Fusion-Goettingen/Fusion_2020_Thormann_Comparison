from constants import *


def get_configs(init_state, ax):
    config_base = {
        'init_state': init_state,
        'init_cov': INIT_STATE_COV,
        'time_steps': TIME_STEPS,
        'ax': ax,
    }

    # ellipse RHMs
    config_ellipseekf_normal = {
        'name': 'Ellipse-RHM-EKF',
        'color': 'red',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'normal',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'normal',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_normal.update(config_base)

    config_ellipseekf_fixed = {
        'name': 'Ellipse-RHM-EKF-fixed',
        'color': 'black',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'normal',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'fixed',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_fixed.update(config_base)

    config_ellipseekf_fixed_c = {
        'name': 'Ellipse-RHM-EKF-fixed_c',
        'color': 'yellow',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'coupled',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'fixed',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_fixed_c.update(config_base)

    config_ellipseekf_fixed_oa = {
        'name': 'Ellipse-RHM-EKF-fixed_oa',
        'color': 'grey',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'normal',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'fixed',  # normal or implicit measurement model
        'al_approx': True,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_fixed_oa.update(config_base)

    config_ellipseekf_normal_coupled = {
        'name': 'Ellipse-RHM-EKF-c',
        'color': 'brown',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'coupled',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'normal',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_normal_coupled.update(config_base)

    config_ellipseekf_imp = {
        'name': 'Ellipse-RHM-EKF-imp',
        'color': 'lime',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),  # in sharp turn, alpha 0.1 in low noise and 0.5 in high noise
        'pred_mode': 'normal',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'imp',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_imp.update(config_base)

    config_ellipseekf_imp_oa = {
        'name': 'Ellipse-RHM-EKF-imp-oa',
        'color': 'yellowgreen',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),
        'pred_mode': 'normal',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'imp',  # normal or implicit measurement model
        'al_approx': True,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_imp_oa.update(config_base)

    config_ellipseekf_imp_coupled = {
        'name': 'Ellipse-RHM-EKF-imp-c',
        'color': 'darkgreen',
        'sigma_q': np.array([1.0, 1.0]),
        'sigma_sh': np.sqrt(np.array([0.01, 0.001, 0.001])),  # in sharp turn and high noise, alpha 0.05
        'pred_mode': 'coupled',  # normal for Cartesian and coupled for polar velocity with single orientation variable
        'mode': 'imp',  # normal or implicit measurement model
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_ellipseekf_imp_coupled.update(config_base)

    # MEM-EKF*
    config_memekfstar = {
        'name': 'MEM-EKF*',
        'color': 'magenta',
        'Q': np.diag([10.0, 10.0, 5.0, 5.0]),
        'SH': np.diag([0.01, 0.001, 0.001]),  # in sharp turn and high noise, alpha 0.05
        'al_approx': False,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_memekfstar.update(config_base)

    config_memekfstar_oa = {
        'name': 'MEM-EKF*-oa',
        'color': 'pink',
        'Q': np.diag([10.0, 10.0, 5.0, 5.0]),
        'SH': np.diag([0.01, 0.001, 0.001]),
        'al_approx': True,  # true for ignoring alpha in state and using orientation of velocity vector instead
    }
    config_memekfstar_oa.update(config_base)

    # IAE
    config_halfaxis = {
        'name': 'IAE',
        'color': 'cyan',
        'sigma_q': np.array([5.0, 5.0]),
        'sigma_sh': np.array([0.01, 0.001, 0.001]),
    }
    config_halfaxis.update(config_base)

    return config_ellipseekf_normal, config_ellipseekf_imp, config_ellipseekf_normal_coupled,\
        config_ellipseekf_imp_coupled, config_ellipseekf_imp_oa, config_memekfstar, config_memekfstar_oa, \
        config_halfaxis, config_ellipseekf_fixed, config_ellipseekf_fixed_c, config_ellipseekf_fixed_oa
