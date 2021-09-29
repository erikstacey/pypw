from models import sin_model, sin_jacobian, n_sin_model, n_sin_jacobian, n_sin_min
from scipy.optimize import curve_fit, dual_annealing
import numpy as np
import matplotlib.pyplot as pl
import lmfit as lm
import config
from Freq import Freq

def sin_mod(x, f, a, p):
    return a*np.sin(2*np.pi*(f*x+p))

def sf_opt_lm(x, data, err, f0, a0, p0):
    sfmod = lm.Model(sin_mod)
    sfmod.set_param_hint("f", value=f0, min=f0*config.freq_bounds_lower_coef, max = f0*config.freq_bounds_upper_coef)
    sfmod.set_param_hint("a", value=a0, min=a0 * config.amp_bounds_lower_coef, max=a0 * config.amp_bounds_upper_coef)
    sfmod.set_param_hint("p", value=p0, min=config.phase_bounds_lower, max=config.phase_bounds_upper)

    sfmod.print_param_hints()

    f_result = sfmod.fit(data=data, weights = err, x=x)
    #print(f_result.fit_report())
    return f_result.best_values["f"], f_result.best_values["a"], f_result.best_values["p"], f_result.best_fit

def mf_opt_lm(x, data, err, freqs):
    sf_mods = []
    for i in range(len(freqs)):
        freq = freqs[i]
        sfmod = lm.Model(sin_mod, prefix=f"f{i}")
        sfmod.set_param_hint(f"f{i}f", value=freq.f, min=freq.f * config.freq_bounds_lower_coef,
                             max=freq.f * config.freq_bounds_upper_coef)
        sfmod.set_param_hint(f"f{i}a", value=freq.a, min=freq.a * config.amp_bounds_lower_coef,
                             max=freq.a * config.amp_bounds_upper_coef)
        sfmod.set_param_hint(f"f{i}p", value=freq.p, min=config.phase_bounds_lower, max=config.phase_bounds_upper)
        sf_mods.append(sfmod)
    mf_mod = np.sum(sf_mods)
    f_result = mf_mod.fit(data=data, weights = err, x=x)
    #print(f_result.fit_report())
    for i in range(len(freqs)):
        freqs[i].f = f_result.best_values[f"f{i}f"]
        freqs[i].a = f_result.best_values[f"f{i}a"]
        freqs[i].p = f_result.best_values[f"f{i}p"]

    return f_result.best_fit






if __name__=="__main__":
    testx = np.linspace(0, 10, 2000)
    testy = sin_mod(testx, 0.5, 2.2, 0.3) + sin_mod(testx, 0.8, 4.2, 0.55) + sin_mod(testx, 0.9, 1.4, 0.11)
    errs = np.ones(len(testx))

    inpt_freqs = [Freq(0.52, 2.3, 0.5, 0), Freq(0.82, 4.4, 0.5, 1), Freq(0.91, 1.5, 0.15, 2)]
    for freq in inpt_freqs:
        freq.prettyprint()
    mf_opt_lm(testx, testy, errs, inpt_freqs)
    for freq in inpt_freqs:
        freq.prettyprint()

