import numpy as np
import matplotlib.pyplot as pl
import config
from optimization import fit_multi_annealing
from pw_io import save_csv_flist
import time as t
import os

def dual_anneal_single(filename_f, filename_data):
    freqs, amps, phases, ferr, aerr, perr, sig = np.loadtxt(filename_f, unpack=True, skiprows=1, delimiter=',')
    time, data, err = np.loadtxt(filename_data, unpack=True)
    if not config.quiet:
        t0 = t.time()
    f_freqs, f_amps, f_phases, zp = fit_multi_annealing(x=time, y=data, err=err, f0=freqs,
                                                    a0=amps, p0=phases, zp0=0, nolocalflag=False)
    if not config.quiet:
        print(f"Dual annealing complete in {t.time()-t0} s")
    save_csv_flist(config.frequencies_da_filename, f_freqs, f_amps, f_phases, ferr, aerr, perr, sig)

if __name__=="__main__":
    os.chdir(config.working_dir)
    dual_anneal_single(config.frequencies_fname, config.target_file)
