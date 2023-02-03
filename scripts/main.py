"""
A program for pre-whitening stellar timeseries.
INPUT:
    A two or three-column time series formatted as [time][delimiter][data][delimiter][uncertainty (if available)]
    If no uncertainty is provided then the data is assumed to have equal weights. The config.py file must be used to
    specify the location of the file, the file, and the parameters of the pre-whitening.

PREPROCESSING:
    Converts data from input type to target type if applicable. Removes a specified number of data points from
    the beginning, end, and around gaps of specified size. Adjusts the JD0.
METHODS:
    Uses a 2-stage iterative pre-whitening method. Dataset object is initialized with an initial light curve object
    (LC0). Each pre-whitening iteration produces a new residual Lightcurve object, which computes its LS periodogram
    when initialized. Each iteration can be described in four stages.

    First Stage/Second Stage: The first stage of each iteration is to identify a candidate frequency/amplitude pair,
    which is used in the second stage to produce an optimized single-frequency sinusoidal model described by a Freq obj.

    Third Stage:
    The third stage simultaneously optimizes frequencies, amplitudes, and phases of each previously-identified Freq obj.
    Fourth Stage:
    The fourth stage subtracts the complete variability model from the LC0 or the single-frequency variability model
    from the residual light curve of the previous iteration to produce a new residual light curve.
OUTPUTS:
    Results are output to a new directory (default "./results/") in the same location as the input file. The file
    structure therein splits into three additional subdirectories - one for auxiliary data and plots of light curves,
    one for auxiliary data and plots of periodograms, and one for supplemental data. The structure is therefore as
    follows (with default names - these can be changed in config):
    /results/
        /lightcurves/
            lc*.png: figures of light curves
            /sf_fits/
                lc_sf*.png: figures of light curves with single-frequency model overplotted
            /mf_fits/
                lc_mf*.png: figures of light curves with complete variability model overplotted
            /data/
                lc*.txt: light curves from each iteration in space-separated files, format [time] [data] [uncert.]
        /periodograms/
            pg*.png: figures of periodograms
            labeled_pg1.png: periodogram with detected frequencies indicated with vertical lines. Zoomed.
            labeled_pg2.png: periodogram with detected frequencies indicated with vertical lines. Full spectrum.
            /slf_fits/
                pg*.png: residual periodograms with SLF fit overplotted
            /lopoly/
                pg*_final_residual: final residual periodogram with low-order polynomial fit overplotted
            /data/
                pg*.txt: periodograms from each iteration in space-separated files, format [frequencies] [amplitudes]
        /supplemental/
            stdevs.png: standard deviation of light curve vs iteration
            misc_data.txt: final slf fit parameters, zp, final std of residual lc

Prerequisite packages:
numpy, matplotlib, astropy, lmfit, scipy (deprecated),

Author: Erik Stacey
Most recently updated Sept 18, 2022

Contact information
Email: erik@erikstacey.com
Alternative : erikwstacey@gmail.com
Github: https://github.com/erikstacey
"""

import os
from config import config
import numpy as np
from preprocessing import preprocess
import matplotlib.pyplot as pl
from pw_io import save_csv_flist, save_config
from Dataset import Dataset

def gen_flists_from_objlist(freqs):
    """
    Convert a list of Freq objects to a list of frequencies, amplitudes, and phases
    :param list freqs: A list of Freq objects
    :return: three lists corresponding to frequencies, amplitudes, and phases
    """
    flist, alist, plist = [], [], []
    for i in range(len(freqs)):
        flist.append(freqs[i].freq)
        alist.append(freqs[i].amp)
        plist.append(freqs[i].phi)
    return flist, alist, plist

def gen_flists_from_objlist_full(freqs):
    """
    Convert a list of Freq objects to a list of frequencies, amplitudes, and phases
    :param list freqs: A list of Freq objects
    :return: four lists corresponding to frequencies, amplitudes, phases, and significance
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
    save_config("configlog.txt")
    os.chdir(config.working_dir)

    # if a third column is specified, it's read as data weights. Otherwise assume equally-weighted data
    if len(config.cols) == 3:
        imptime, impdata, imperr = np.loadtxt(fname=config.target_file, usecols=config.cols,
                                              unpack=True, delimiter = config.delimiter,
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

    # show a plot of the pre-processed data before proceeding
    pl.plot(pptime, ppdata, color='black', marker='.', markersize=0.5, linestyle=None)
    pl.xlabel("Time [HJD]")
    pl.ylabel(f"Amplitude [{config.target_dtype}]")
    pl.show()

    # dataset object stores all the data and executes pre-whitening iterations
    ds = Dataset(pptime, ppdata, pperr, reference_flux=ref_flux)

    # pre-whiten until ds.it_pw() indicates to terminate the analysis, then do post-processing
    while len(ds.freqs) < config.n_f:
        c_code = ds.it_pw()
        if c_code == 1:
            break
    ds.post()


main()
