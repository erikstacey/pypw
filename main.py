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
Most recently updated Aug 08 2022
Contact information:
Email: erik@erikstacey.com
Alternative : erikwstacey@gmail.com
"""

import os
import sys
import shutil
import config
import numpy as np
from preprocessing import preprocess
from Lightcurve import Lightcurve
from Freq import Freq
import matplotlib.pyplot as pl
import time
from pw_io import save_csv_flist, save_config
from Dataset import Dataset

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


pl.rcParams['figure.figsize'] = [5, 3.5]
pl.rcParams['figure.dpi'] = 300

pl.rcParams['font.family'] = ['serif'] # default is sans-serif
pl.rcParams['font.serif'] = [
           'Times New Roman',
           'Times',
           'Bitstream Vera Serif',
           'DejaVu Serif',
           'New Century Schoolbook',
           'Century Schoolbook L',
           'Utopia',
           'ITC Bookman',
           'Bookman',
           'Nimbus Roman No9 L',
           'Palatino',
           'Charter',
           'serif']

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 11}

pl.rc('font', **font)

def main():
    # todo: use low snr cutoff and high iteration cutoff to extract lots of frequencies. Then assess them individually.
    save_config("configlog.txt")
    os.chdir(config.working_dir)
    if len(config.cols) == 3:
        imptime, impdata, imperr = np.loadtxt(fname=config.target_file, usecols=config.cols, unpack=True, delimiter = config.delimiter,
                                              skiprows=config.skiprows)

    elif len(config.cols) == 2:
        imptime, impdata = np.loadtxt(fname=config.target_file, usecols=config.cols, unpack=True,
                                              delimiter=config.delimiter,
                                      skiprows=config.skiprows)
        imperr = np.ones(len(imptime))
    imptime += config.file_time_offset
    pptime, ppdata, pperr, ref_flux = preprocess(imptime-config.jd0, impdata, imperr)
    print(f"Reference flux set to: {ref_flux}")
    print(f"JD0 = {config.jd0}")
    print(f"Midpoint: {(pptime[-1]-pptime[0])/2}")
    pl.plot(pptime, ppdata, color='black', marker='.', markersize=0.5, linestyle=None)
    pl.xlabel("Time [HJD]")
    pl.ylabel(f"Amplitude [{config.target_dtype}]")
    pl.show()

    ds = Dataset(pptime, ppdata, pperr, reference_flux=ref_flux)

    while len(ds.freqs) < config.n_f:
        c_code = ds.it_pw()
        if c_code == 1:
            break
    ds.post()
    if not config.quiet:
        print("Saving results...")
    if not config.quiet:
        print("Done!")


main()
