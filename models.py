import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl

def chisq(data, model, err):
    return sum(((data-model)/err)**2)

def sin_model(x, freq, amp, phase):
    return amp*np.sin(2*np.pi*(freq*x+phase))

def sin_jacobian(x, cf, ca, cp, flag=None):
    """ An n x m matrix corresponding to the jacobian for a single 3-parameter sinusoid.
    n = number of data pts
    m = 1, 2, 3 depending on fixed parameters"""
    jacobian = np.zeros((len(x), 3))
    jacobian[:, 0] = 2 * np.pi * ca * x * np.cos(2 * np.pi * (cf * x + cp))
    jacobian[:, 1] = np.sin(2 * np.pi * (cf * x + cp))
    jacobian[:, 2] = 2 * np.pi * ca * np.cos(2 * np.pi * (cf * x + cp))
    if flag == "fixed_fa":
        ret_jac = jacobian[:, 2:]
        return ret_jac
    elif flag == "fixed_f":
        ret_jac = jacobian[:, 1:]
        return ret_jac
    else:
        return jacobian


def n_sin_model(x, *params):
    """params must be in format of *freqs, *amps, *phases"""
    l = len(params)
    nfreqs = len(params)//3
    y = np.zeros(len(x))
    for i in range(l // 3):
        y += sin_model(x, params[i], params[i+nfreqs], params[i+2*nfreqs])
    return y

def n_sin_jacobian(x, *params, flag=None):
    """ returns jacobian of the n_sin model above. Can use flag to control the shape of the output for fixing params
     flags: fixed_fa; just includes phase components, fixed_f; just includes amplitude and phase components,"""
    jacobian = np.zeros((len(x), len(params)))
    n_f = int((len(params)-1) / 3)
    for k in range(n_f):
        cf = params[k]
        ca = params[n_f+k]
        cp = params[2*n_f+k]
        jacobian[:, k] = 2*np.pi*ca*x*np.cos(2*np.pi*(cf*x+cp))
        jacobian[:, n_f+k] = np.sin(2*np.pi*(cf*x+cp))
        jacobian[:, 2*n_f+k] = 2*np.pi*ca*np.cos(2*np.pi*(cf*x+cp))
    if flag == "fixed_fa":
        ret_jac = jacobian[:, 2*n_f:]
        return ret_jac
    elif flag == "fixed_f":
        ret_jac = jacobian[:, n_f:]
        return ret_jac
    else:
        return jacobian

def n_model_poly(x, *params):
    power = 0
    try:
        out = np.zeros(len(x))
    except TypeError:
        out = 0
    for p in params:
        out += p * (x**power)
        power+=1
    return out

def n_sin_min(x, y, err, *params):
    model = n_sin_model(x, *params)
    dof = len(x)-len(params)
    return chisq(y, model, err)/dof

def bowman_noise_model(x, *params):
    # params = [x0, alpha_0, gamma, Cw]
    return params[1] / (1+(x/params[0])**params[2]) + params[3]

if __name__ == "__main__":
    x = np.linspace(0,30, 10000)
