from models import sin_model, chisq
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import config

def min_n_sin_model(x, pars):
    """params must be in format of *freqs, *amps, *phases"""
    for i in range(l // 3):
        y += sin_model(x, params[i], params[i+nfreqs], params[i+2*nfreqs])
    return y

def fit_multi_lmfit(x, y, err, f0, a0, p0):
    fit_p = lm.Parameters()
    for i in range(len(f0)):
        fit_p.add(f"f{i}", value = f0[i],
                  min=config.lm_freq_bounds_lower_coef*f0[i], max = config.lm_freq_bounds_upper_coef*f0[i])
        fit_p.add(f"a{i}", value = a0[i],
                  min=config.lm_amp_bounds_lower_coef*f0[i], max = config.lm_amp_bounds_upper_coef*f0[i])
        fit_p.add(f"p{i}", value = p0[i],
                  min=config.lm_phase_bounds_lower, max = config.lm_phase_bounds_upper)


if __name__ == "__main__":
    xtest = np.linspace(0, 30, 10000)
    ytest = np.ones(len(xtest))
    errs = np.ones(len(xtest))
    f0 = [1.64, 0.82, 1.2, 2.5]
    a0 = [17, 11, 6, 4]
    p0 = [0.8, 0.98, 0.6, 0.1]
    fit_multi_lmfit(x=xtest, y = ytest, err= errs, f0=f0, a0=a0, p0=p0)