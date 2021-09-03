from models import sin_model, chisq
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import config

def min_n_sin_model(params, x, data, err):
    """params must be in format of *freqs, *amps, *phases"""
    n_f = len(params)//3
    model = np.zeros(len(x))
    for i in range(n_f):
        model += sin_model(x, params[f"f{i}"], params[f"a{i}"], params[f"p{i}"])
    return (model-data) / err

def fit_multi_lmfit(x, data, err, f0, a0, p0):
    fit_p = lm.Parameters()
    for i in range(len(f0)):
        fit_p.add(f"f{i}", value = f0[i],
                  min=config.lm_freq_bounds_lower_coef*f0[i], max = config.lm_freq_bounds_upper_coef*f0[i])
        fit_p.add(f"a{i}", value = a0[i],
                  min=config.lm_amp_bounds_lower_coef*a0[i], max = config.lm_amp_bounds_upper_coef*a0[i])
        fit_p.add(f"p{i}", value = p0[i],
                  min=config.lm_phase_bounds_lower, max = config.lm_phase_bounds_upper)
    minner = lm.Minimizer(min_n_sin_model, fit_p, fcn_args=(x, data, err))
    min_res = minner.minimize()
    min_pardict = min_res.params
    out_f = np.zeros(len(f0))
    out_a = np.zeros(len(f0))
    out_p = np.zeros(len(f0))
    for key in min_pardict.keys():
        if key[0] == "f":
            out_f[int(key[1])] = min_pardict[key]
        elif key[0] == "a":
            out_a[int(key[1])] = min_pardict[key]
        elif key[0] == "p":
            out_p[int(key[1])] = min_pardict[key]
    return out_f, out_a, out_p



if __name__ == "__main__":
    xtest = np.linspace(0, 30, 10000)
    ytest = np.zeros(len(xtest))
    errs = np.ones(len(xtest))
    f0_true = [1.64, 0.82, 1.2, 2.5]
    a0_true = [17, 11, 6, 4]
    p0_true = [0.8, 0.98, 0.6, 0.1]
    for i in range(len(f0_true)):
        ytest += sin_model(xtest, f0_true[i], a0_true[i], p0_true[i])
    f0 = [1.63, 0.81, 1.21, 2.55]
    a0 = [16.5, 11.4, 7, 4.2]
    p0 = [0.88, 1.02, 0.9, 0.2]
    fit_f, fit_a, fit_p = fit_multi_lmfit(x=xtest, data = ytest, err= errs, f0=f0, a0=a0, p0=p0)

    model_y = np.zeros(len(xtest))
    for i in range(len(fit_f)):
        model_y += sin_model(xtest, fit_f[i], fit_a[i], fit_p[i])
        print(fit_f[i], fit_a[i], fit_p[i])
    pl.plot(xtest, ytest)
    pl.plot(xtest, model_y)
    pl.show()
