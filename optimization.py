from models import sin_model, sin_jacobian, n_sin_model, n_sin_jacobian, n_sin_min
from scipy.optimize import curve_fit, dual_annealing
import numpy as np
import matplotlib.pyplot as pl
import config
def unpack_paramlist(params):
    # converts a list of [*freqs, *amps, *phases, zp] into 3 arrays
    c_n_f = int((len(params))/3)
    freqs = params[:c_n_f]
    amps = params[c_n_f:2 * c_n_f]
    phases = params[2 * c_n_f:]
    return freqs, amps, phases

def make_lm_bounds(f0, a0, p0):
    out_bounds = [[],[]]
    for i in range(len(f0)*3):
        out_bounds[0].append(-np.inf)
        out_bounds[1].append(np.inf)
    if config.lm_freq_bounds:
        for i in range(len(f0)):
            out_bounds[0][i] = config.lm_freq_bounds_lower_coef*f0[i]
            out_bounds[1][i] = config.lm_freq_bounds_upper_coef*f0[i]
    if config.lm_amp_bounds:
        for i in range(len(f0)):
            out_bounds[0][i+len(f0)] = config.lm_amp_bounds_lower_coef * a0[i]
            out_bounds[1][i+len(f0)] = config.lm_amp_bounds_upper_coef * a0[i]
    if config.lm_phase_bounds:
        for i in range(len(f0)):
            out_bounds[0][i+2*len(f0)] = config.lm_phase_bounds_lower
            out_bounds[1][i+2*len(f0)] = config.lm_phase_bounds_upper
    return out_bounds

def fit_single_lm(x, y, err, f0, a0, p0):
    p0 = [f0, a0, p0]
    p, _ = curve_fit(f=sin_model, xdata=x, ydata=y, sigma=err, p0=p0,)#jac=sin_jacobian_zp)
    return p[0], p[1], p[2]

def check_bounds(parameters, bounds):
    for i in range(len(parameters)):
        if abs(parameters[i] - bounds[0][i]) < parameters[i]*0.05 or abs(parameters[i] - bounds[1][i]) < parameters[i]*0.05:
            print(f"\t\tParameter {i} ({parameters[i]} within 5% of a bound ({[bounds[0][i], bounds[1][i]]})")

def fit_multi_lm(x, y, err, f0, a0, p0, nolocalflag=None):
    # fits n sinusoids together as defined by f0, a0, p0
    n_f = len(f0)
    p = [*f0, *a0, *p0]
    bounds = make_lm_bounds(f0, a0, p0)

    if config.fixed_freq_multi:
        p_f = p[:n_f]
        p_nof = p[n_f:]
        fixed_freq_f = lambda x, *nofpar : n_sin_model(x, *p_f, *nofpar)
        #fixed_freq_jac = lambda x, *fpar : n_sin_jacobian(x, *p_f, *p_nof, flag="fixed_f")
        bounds_ff = [bounds[0][n_f:], bounds[1][n_f:]]
        p_nof, _ = curve_fit(f=fixed_freq_f, xdata=x, ydata=y, sigma=err, p0=p_nof, bounds=bounds_ff)
        p[n_f:] = p_nof
    if config.fixed_none_multi:
        p, _ = curve_fit(f=n_sin_model, xdata=x, ydata=y, sigma=err, p0=p, bounds=bounds)
        check_bounds(p, bounds)
    freqs, amps, phases = unpack_paramlist(p)
    return freqs, amps, phases

def fit_multi_annealing(x, y, err, f0, a0, p0, nolocalflag=True):
    # fits n sinusoids as defined by f0, a0, p0
    # uses classical simulated annealing
    print("Starting annealing...")
    p0 = [*f0, *a0, *p0]
    anneal_f = lambda f_p : n_sin_min(x, y, err, *f_p)
    # set bounds for each parameter
    f_b, a_b, p_b = [], [], []
    if config.ann_bounds_method == "relative":
        for i in range(len(f0)):
            f_b.append((f0[i]*(1-config.ann_relfrac_freq), f0[i]*(1+config.ann_relfrac_freq)))
            a_b.append((a0[i] * (1 - config.ann_relfrac_amp), a0[i] * (1 + config.ann_relfrac_amp)))
            p_b.append((p0[i] * (1 - config.ann_relfrac_phase), p0[i] * (1 + config.ann_relfrac_phase)))
    bounds = [*f_b, *a_b, *p_b, (config.ann_zp_bounds_lower, config.ann_zp_bounds_upper)]
    ann_res = dual_annealing(func=anneal_f, bounds=bounds, no_local_search=nolocalflag, x0= p0)
    p = ann_res.x
    if not config.quiet:
        print(f"Anneal successful? {ann_res.success} in {ann_res.nit} iterations and {ann_res.nfev} fn evals")
        print(f"Anneal result status: {ann_res.status}")
        print(f"Anneal result msg: {ann_res.message}")

    ret_f, ret_a, ret_p = unpack_paramlist(p)
    return ret_f, ret_a, ret_p
if __name__=="__main__":
    print(make_lm_bounds([1.1, 1.5, 1.6], [10, 7, 3], [0.2, 0.3, 0.9]))