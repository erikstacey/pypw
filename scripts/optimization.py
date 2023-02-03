import numpy as np
import lmfit as lm
from config import config
from Freq import Freq

def constant_function(x, z):
    return z

def sin_mod(x, f, a, p):
    return a*np.sin(2*np.pi*(f*x+p))

def sf_opt_lm(x, data, err, f0, a0, p0):
    """Optimize a single-frequency sinusoidal model using lmfit"""
    sfmod = lm.Model(sin_mod)
    sfmod.set_param_hint("f", value=f0, min=f0*config.freq_bounds_lower_coef, max = f0*config.freq_bounds_upper_coef)
    sfmod.set_param_hint("a", value=a0, min=a0 * config.amp_bounds_lower_coef, max=a0 * config.amp_bounds_upper_coef)
    sfmod.set_param_hint("p", value=p0, min=config.phase_bounds_lower, max=config.phase_bounds_upper)

    sfmod.print_param_hints()

    f_result = sfmod.fit(data=data, weights = err, x=x)
    #print(f_result.fit_report())
    return f_result.best_values["f"], f_result.best_values["a"], f_result.best_values["p"], f_result.best_fit


def mf_opt_lm(x, data, err, freqs, zp):
    """
    Optimize a multi-frequency model (complete variability model) using LMFit
    :param x: The time axis of the original light curve
    :param data: The data axis of the original light curve
    :param err: The uncertainties on the data of the original light curve
    :param freqs: A list of Freq objects passed by reference. These will be updated in-place.
    :param zp: The floating-mean zp from the previous multi-frequency optimization
    :return: A best-fit model evaluated at the input x values, a new optimized floating-mean zp
    """
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
    zeroptmodel = lm.Model(constant_function, prefix="zp")
    zeroptmodel.set_param_hint(f"zpz", value=zp)
    if config.multi_fit_zp:
        sf_mods.append(zeroptmodel)
    mf_mod = np.sum(sf_mods)
    f_result = mf_mod.fit(data=data, weights = err, x=x)
    #print(f_result.fit_report())

    for i in range(len(freqs)):
        # check boundaries - If any parameters are suspiciously close to a boundary, warn the user
        c_sfmod = sf_mods[i]
        for ptype in ["f", "a", "p"]:
            if abs(c_sfmod.param_hints[ptype]["min"]-f_result.best_values[f"f{i}{ptype}"]) \
                    < config.boundary_warnings * f_result.best_values[f"f{i}{ptype}"]:
                print(f"\t\t WARNING: {ptype} {f_result.best_values[f'f{i}{ptype}']} of f{i} within"
                      f" {config.boundary_warnings*100}% of lower boundary {c_sfmod.param_hints[ptype]['min']}")
            elif abs(c_sfmod.param_hints[ptype]["max"]-f_result.best_values[f"f{i}{ptype}"]) <\
                    config.boundary_warnings * f_result.best_values[f"f{i}{ptype}"]:
                print(f"\t\t WARNING: {ptype} {f_result.best_values[f'f{i}{ptype}']} of f{i}"
                      f" within {config.boundary_warnings*100}% of upper boundary {c_sfmod.param_hints[ptype]['max']}")
        # update the frequencies in-place
        freqs[i].f = f_result.best_values[f"f{i}f"]
        freqs[i].a = f_result.best_values[f"f{i}a"]
        freqs[i].p = f_result.best_values[f"f{i}p"]
    if config.multi_fit_zp:
        print(f"MF ZP: {f_result.best_values['zpz']}")
    return f_result.best_fit, f_result.best_values["zpz"]






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

