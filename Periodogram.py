import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from models import n_model_poly
from astropy.timeseries import LombScargle
import config

class Periodogram():
    """Computes and stores a Lomb-Scargle periodogram"""
    def __init__(self, time, data):
        """time, data are numpy arrays representing a timeseries, and then lsfreq and lsamp will represent the
        power spectrum corresponding to the supplied timeseries. lsamp has units of the amplitude unit of the
        timeseries unless an error is specified, in which case lsamp has no units"""

        self.resolution = 1.5/(max(time)-min(time))
        if config.periodograms_lowerbound == "resolution":
            lowerbound = self.resolution
        else:
            lowerbound = config.periodograms_lowerbound
        upperbound = config.periodograms_upperbound

        lsfreq = np.linspace(lowerbound, upperbound, 10 * int((upperbound - lowerbound) / self.resolution))
        # lsfreq = np.linspace(0.001, 20, 20000)

        lspower = LombScargle(time, data, normalization='psd').power(lsfreq)

        N = len(time)
        lsamp = 2 * (abs(lspower) / N) ** 0.5


        self.lsfreq = lsfreq
        self.lsamp = lsamp
        self.polyfunc = n_model_poly
        self.polyfit = None
        self.polyparams_log = None
        self.logx = None
        self.logy = None
    def highest_ampl(self, mask=None):
        """ Return the frequency corresponding to the highest amplitude in the amplitude spectrum, and the amplitude at
        that frequency"""
        if mask is None:
            ymax = np.argmax(self.lsamp)
            xatpeak = self.lsfreq[ymax]
        else:
            filtered_lsamp = self.lsamp[mask]
            filtered_lsfreq = self.lsfreq[mask]
            ymax = np.argmax(filtered_lsamp)
            xatpeak = filtered_lsfreq[ymax]
        return xatpeak, self.lsamp[ymax]
    def get_avg_around_pt(self, freq, width):
        indices_to_use = []
        for i in range(len(self.lsfreq)):
            if freq-width <= self.lsfreq[i] < freq+width:
                indices_to_use.append(i)
        return np.mean(self.lsamp[indices_to_use])

    def peak_selection_w_bins(self):
        selection_exclusion_mask = np.ones(len(self.lsfreq), dtype=bool)
        for it in range(config.cutoff_iteration):
            cf, ca = self.highest_ampl(mask = selection_exclusion_mask)
            loc_avg = self.get_avg_around_pt(cf, config.averaging_bin_width)
            sig = ca/loc_avg
            if sig > config.cutoff_sig:
                return cf, ca
            else:
                freq_spacing = self.lsfreq[1]-self.lsfreq[0]
                index_excl_radius = int(self.resolution // freq_spacing)
                central_i = np.where(self.lsfreq == cf)[0][0]
                lower_i = 0 if central_i - index_excl_radius < 0 else central_i-index_excl_radius
                upper_i = len(self.lsfreq) if central_i+index_excl_radius > len(self.lsfreq) else central_i+index_excl_radius
                selection_exclusion_mask[lower_i:upper_i] = False
        return None, None

    def fit_self_lopoly(self):
        p0 = [0, 0, 0, 0]
        self.logx = np.log10(self.lsfreq)
        self.logy = np.log10(self.lsamp)
        p, _ = curve_fit(self.polyfunc, self.logx, self.logy, p0)
        self.polyparams_log = p
        self.polyfit = 10**self.polyfunc(self.logx, *p)
        print(f"Complete fit of LO poly to log log periodogram")
        print(f"params = {p}")


    def get_polyfit_at_val(self, frequency):
        logfreq = np.log10(frequency)
        logval = self.polyfunc(logfreq, *self.polyparams_log)
        return 10**logval

    def plot_polyfit_log(self, savename=None):
        pl.plot(self.logx, self.logy, color='black', label='data')
        pl.plot(self.logx, self.polyfunc(self.logx, *self.polyparams_log), color='red', label='order 3 poly fit')
        pl.xlabel("Freq [c/d]")
        pl.ylabel("log(Amp [mmag])")
        if not savename:
            pl.show()
            pl.clf()
        else:
            pl.savefig(savename)
            pl.clf()
    def plot_polyfit_normspace(self, savename=None):
        pl.plot(self.lsfreq, self.lsamp, color='black', label='data')
        pl.plot(self.lsfreq, self.polyfit, color='red', label='fit')
        pl.legend()
        pl.xlabel("Freq [c/d]")
        pl.ylabel("Amp [mmag]")
        if not savename:
            pl.show()
            pl.clf()
        else:
            pl.savefig(savename)
            pl.clf()


    def diagnostic_self_plot(self, vline = None, show=False, savename = None):
        pl.plot(self.lsfreq, self.lsamp, color='black')
        pl.xlim(0, 2)
        if vline is not None:
            pl.axvline(vline,color='red')
        if show:
            pl.show()
            pl.clf()
        elif savename:
            pl.savefig(savename)
            pl.clf()
