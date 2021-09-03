"""
A program for pre-whitening stellar timeseries.
INPUT:
    A 3-column file with columns corresponding to time, data, and error values. The units can be in flux, mag, or mmag.
    A config.py file in the directory - see included for notes and instructions.
PREPROCESSING:
    Converts data from input type to target type if applicable. Also removes a specified number of data points from
    the beginning, end, and around gaps of specified size.
METHODS:
    Uses a 2-stage iterative pre-whitening method. First, a periodogram is computed and an initial frequency and
    amplitude are identified. A single-frequency fit (LM) is performed to the current residual periodogram LC with all
    the frequency values fixed (2*n+1 parameters) and then with all parameters free (3n+1 parameters). This can be done
    with an LM algorithm or by classical simulated annealing. A residual light curve is computed by subtracting this
    full model from the original light curve.
OUTPUTS:
    All outputs are to the "working directory" specified in the config file.
    "configlog.txt": a snapshot of the config file at the time the program was run
    "freq_logs/*.csv": frequency lists at each iterative stage
    "frequencies.csv": the final list of frequency results

Author: Erik Stacey
Most recently updated Aug 25, 2021
Contact information:
Email: erik@erikstacey.com
Alternative : erikwstacey@gmail.com
"""

import os
import sys
import config
import numpy as np
from preprocessing import preprocess
from Lightcurve import Lightcurve
from Freq import Freq
import matplotlib.pyplot as pl
import time
from pw_io import save_csv_flist, save_config
from models import sin_model

from optimization import fit_multi_lm, fit_single_lm, fit_multi_annealing
from optimization_lm import fit_multi_lmfit

def gen_flists_from_objlist(freqs):
    """Takes list of frequency objects (Freq.py) and converts it to a list of frequency, amplitude, and phase vals"""
    flist, alist, plist = [], [], []
    for i in range(len(freqs)):
        flist.append(freqs[i].freq)
        alist.append(freqs[i].amp)
        plist.append(freqs[i].phi)
    return flist, alist, plist

def gen_flists_from_objlist_full(freqs):
    """Takes list of frequency objects (Freq.py) and converts it to a list of frequency, amplitude, phase, and sig vals
    """
    flist, alist, plist, slist = [], [], [], []
    for i in range(len(freqs)):
        flist.append(freqs[i].freq)
        alist.append(freqs[i].amp)
        plist.append(freqs[i].phi)
        slist.append(freqs[i].sig)
    return flist, alist, plist, slist

def main():
    save_config("configlog.txt")
    os.chdir(config.working_dir)
    imptime, impdata, imperr = np.loadtxt(fname=config.target_file, usecols=config.cols, unpack=True)
    pptime, ppdata, pperr = preprocess(imptime, impdata, imperr)
    pl.plot(pptime, ppdata)
    pl.show()
    LC0 = Lightcurve(time=pptime, data=ppdata, err=pperr)
    LCs = np.array([LC0], dtype=Lightcurve)
    freqs = np.array([], dtype=Freq)
    mf_model = np.zeros(len(LC0.time), dtype=float)
    res = 1.5/(max(LC0.time)-min(LC0.time))
    print(f"Performing analysis assuming frequency resolution of {res:.3f}")
    if config.save_freq_logs:
        os.makedirs(config.working_dir+"/"+config.freqlog_folder, exist_ok=True)

    if not config.quiet:
        print("Starting iterations")
    for i in range(config.n_f):
        if not config.quiet:
            print(f"===== STARTING ITERATION {i} =====")
        ##### Stage 1: Identify frequency on newest LC and perform SF fit
        if not config.quiet:
            print("\tStarting stage 1 - identify new frequency and perform single fit")
            ct0 = time.time()
        if config.freq_selection_method == "highest":
            cpg_f, cpg_a = LCs[-1].periodogram.highest_ampl()
        elif config.freq_selection_method == "binned":
            cpg_f, cpg_a = LCs[-1].periodogram.peak_selection_w_bins()
        if cpg_f is None:
            print(f"Stop criterion reached at iteration {i}")
            break
        c_p_guess = 0
        while True:
            sf_f, sf_a, sf_p = fit_single_lm(LCs[-1].time, LCs[-1].data, LCs[-1].err, cpg_f, cpg_a, c_p_guess)
            if abs(sf_p-c_p_guess) > config.phase_fit_rejection_criterion:
                break
            else:
                c_p_guess += 0.4
        sf_model = sin_model(LC0.time, sf_f, sf_a, sf_p)
        if config.plot_iterative_lcs:
            pl.plot(LCs[-1].time, LCs[-1].data, linestyle='none', marker='.', markersize=1, color="black")
            pl.plot(LCs[-1].time, sin_model(LCs[-1].time, sf_f, sf_a, sf_p), color='red')
            pl.savefig(f"figures_lcs_iterative/sf_model_{i}.png")
            pl.clf()
        freqs = np.append(freqs, Freq(sf_f, sf_a, sf_p, n=i))
        freqs[-1].prettyprint()
        ##### Stage 2: unpack frequencies from list of objects to freq, amp, phase arrays
        if not config.quiet:
            print(f"\tStage 1 complete in {time.time()-ct0}")
            print("\tStarting stage 2 - setting up for multi-frequency fit")
            ct0 = time.time()
        mf_freqs0, mf_amps0, mf_phases0 = gen_flists_from_objlist(freqs)
        ##### Stage 3: multi-frequency optimization
        if not config.quiet:
            print(f"\tStage 2 complete in {time.time()-ct0}")
            print("\tStarting stage 3 - multi-frequency fit")
            ct0 = time.time()
        if config.multi_fit_type=="lm":
            fit_multi_fn = fit_multi_lmfit
        elif config.multi_fit_type=="anneal":
            fit_multi_fn = fit_multi_annealing
        elif config.multi_fit_type == "scipy":
            fit_multi_fn = fit_multi_lm
        fit_freqs, fit_amps, fit_phases = fit_multi_fn(x=LC0.time, y=LC0.data, err=LC0.err, f0=mf_freqs0,
                                                            a0=mf_amps0, p0=mf_phases0)
        ##### Stage 4: update frequency objects
        if not config.quiet:
            print(f"\tStage 3 complete in {time.time()-ct0}")
            print("\tStarting stage 4 - updating frequencies")
            ct0 = time.time()
        for k in range(len(fit_freqs)):
            freqs[k].update(fit_freqs[k], fit_amps[k], fit_phases[k])
        #### Stage 5: (re)generate current model
        # TODO: try subtracting off single-frequency models from residuals and see if that helps
        if not config.quiet:
            print(f"\tStage 4 complete in {time.time()-ct0}")
            print("\tStarting stage 5 - making current model")
            ct0 = time.time()

        mf_model[:] = 0
        for freq in freqs:
            mf_model += freq.genmodel(LC0.time)

        ##### Stage 6: Create and store residual light curve
        if not config.quiet:
            print(f"\tStage 5 complete in {time.time()-ct0}")
            print("\tStarting stage 6 - making and storing residual LC")
            ct0 = time.time()
        if config.residual_model_generation == "sf":
            r_lc = Lightcurve(LC0.time, LCs[-1].data-sf_model, LC0.err)
        elif config.residual_model_generation == "mf":
            r_lc = Lightcurve(LC0.time, LCs[0].data - mf_model, LC0.err)
        if config.plot_iterative_lcs:
            r_lc.diag_plot(show=False, savename=f"figures_lcs_iterative/residual_{i}.png")
        if config.plot_iterative_pgs:
            r_lc.periodogram.diagnostic_self_plot(savename=f"figures_periodograms_iterative/{i}.png")
        LCs = np.append(LCs, r_lc)
        if not config.quiet:
            print(f"\tStage 6 complete in {time.time()-ct0}")
            print("CURRENT RESULTS")
            for freq in freqs:
                freq.prettyprint()
        if config.save_freq_logs:
            sfreqs, samps, sphases = gen_flists_from_objlist(freqs)
            save_csv_flist(config.freqlog_folder+f"/freqs_{i}.csv", sfreqs, samps, sphases, np.zeros(len(sfreqs)),
                           np.zeros(len(sfreqs)),
                           np.zeros(len(sfreqs)), np.zeros(len(sfreqs)))

        if not config.quiet:
            print("\n\n")
    # post events
    final_pg = LCs[-1].periodogram
    final_pg.fit_self_lopoly()
    final_pg.plot_polyfit_log()
    final_pg.plot_polyfit_normspace()
    for freq in freqs:
        freq.sig = freq.amp / final_pg.get_polyfit_at_val(freq.freq)
        freq.prettyprint_sig()

    sfreqs, samps, sphases, ssigs = gen_flists_from_objlist_full(freqs)
    save_csv_flist(config.frequencies_fname, sfreqs, samps, sphases, ssigs,np.zeros(len(sfreqs)),
                   np.zeros(len(sfreqs)), np.zeros(len(sfreqs)))

main()
