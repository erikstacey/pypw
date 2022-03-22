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

        self.resolution = 1.5 / (max(time) - min(time))
        if config.periodograms_lowerbound == "resolution":
            lowerbound = self.resolution
        else:
            lowerbound = config.periodograms_lowerbound
        upperbound = config.periodograms_upperbound

        lsfreq = np.linspace(lowerbound, upperbound, 50 * int((upperbound - lowerbound) / self.resolution))
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
            return xatpeak, self.lsamp[ymax]
        else:
            filtered_lsamp = self.lsamp[mask]
            filtered_lsfreq = self.lsfreq[mask]
            ymax = np.argmax(filtered_lsamp)
            xatpeak = filtered_lsfreq[ymax]
            return xatpeak, filtered_lsamp[ymax]

    def find_troughs(self, center):
        count = 2
        left_i, right_i = None, None
        while (not left_i) or (not right_i):
            count += 1
            # check for left trough
            if not left_i and (center - count == 0 or self.lsamp[center - count] >= self.lsamp[center - count + 1]):
                left_i = center - count + 1
            if not right_i and (
                    center + count == len(self.lsfreq) or self.lsamp[center + count] >= self.lsamp[center + count - 1]):
                right_i = center + count - 1
        if left_i < 0:  # ensure no wraparound
            left_i = 0
        return left_i, right_i

    def find_index_of_freq(self, t):
        c_arr = self.lsfreq
        lower_bound = 0
        upper_bound = len(c_arr)
        while True:
            mid_i = (upper_bound - lower_bound) // 2 + lower_bound
            if t > self.lsfreq[mid_i]:
                lower_bound = mid_i
            elif t <= self.lsfreq[mid_i]:
                upper_bound = mid_i
            if upper_bound - lower_bound <= 1:
                lower_diff = abs(self.lsfreq[lower_bound] - t)
                upper_diff = abs(self.lsfreq[upper_bound] - t)
                if lower_diff > upper_diff:
                    return upper_bound
                else:
                    return lower_bound

                return lower_bound

    def get_peak_sig(self, center_val_freq, freq_amp):
        center_i_freq = np.where(self.lsfreq == center_val_freq)[0][0]
        trough_left_i, trough_right_i = self.find_troughs(center_i_freq)
        if not config.quiet:
            print(f"\tPeak spans {self.lsfreq[trough_left_i]:.3f} to {self.lsfreq[trough_right_i]}")
        lower_val_freq = self.lsfreq[center_i_freq] - config.averaging_bin_radius
        upper_val_freq = self.lsfreq[center_i_freq] + config.averaging_bin_radius
        if lower_val_freq < 0:
            lower_val_freq = 0
        if upper_val_freq > config.periodograms_upperbound:
            upper_val_freq = config.periodograms_upperbound
        lower_i_freq = self.find_index_of_freq(lower_val_freq)
        upper_i_freq = self.find_index_of_freq(upper_val_freq)
        if lower_i_freq < 0:
            lower_i_freq = 0
        if upper_i_freq > len(self.lsfreq):
            upper_i_freq = len(self.lsfreq)

        lower_avg_region = self.lsamp[lower_i_freq:trough_left_i]
        upper_avg_region = self.lsamp[trough_right_i:upper_i_freq]
        total_avg_region = np.concatenate((lower_avg_region, upper_avg_region))

        avg_regions_avg = np.mean(total_avg_region)
        if not config.quiet:
            print(
                f"\tAveraged from {lower_val_freq:.3f}:{self.lsfreq[trough_left_i]:.3f} and {self.lsfreq[trough_right_i]:.3f}:{upper_val_freq:.3f}")
            print(f"\tYielded avg = {avg_regions_avg:.3f} ||| Nom. Freq. amp = {self.lsamp[center_i_freq]:.3f}")
        if config.runtime_plots:
            pl.plot(self.lsfreq, self.lsamp, color='black')
            pl.plot(self.lsfreq[lower_i_freq:trough_left_i], lower_avg_region, color='orange')
            pl.plot(self.lsfreq[trough_right_i:upper_i_freq], upper_avg_region, color='orange')
            pl.axvline(lower_val_freq, color='blue')
            pl.axvline(upper_val_freq, color='blue')
            pl.axvline(self.lsfreq[trough_left_i], color='red')
            pl.axvline(self.lsfreq[trough_right_i], color='red')
            pl.axvline(center_val_freq, color='black', linestyle='--')
            pl.xlim(0, 4)
            pl.ylim(0, 20)
            pl.show()
            pl.clf()

        return freq_amp / avg_regions_avg, trough_left_i, trough_right_i

    def get_sig_boxavg(self, center_val_freq, freq_amp):
        """ This gets significance of a frequency with f=center_val_freq and A=freq_amp without considering
        a peak width - this is used to assess frequency significance after all frequencies have been identified
        using the final residual periodogram"""
        center_i_freq = np.where(self.lsfreq == center_val_freq)[0]
        # find values of edge frequencies
        lower_val_freq = self.lsfreq[center_i_freq] - config.averaging_bin_radius
        upper_val_freq = self.lsfreq[center_i_freq] + config.averaging_bin_radius
        # find left and right indices
        lower_i_freq = self.find_index_of_freq(lower_val_freq)
        upper_i_freq = self.find_index_of_freq(upper_val_freq)
        total_avg_region = self.lsamp[lower_i_freq:upper_i_freq]
        return freq_amp / np.mean(total_avg_region)

    def peak_selection_w_sig(self):
        cur_mask = np.ones(len(self.lsfreq), dtype=bool)
        count = 0
        while count < config.cutoff_iteration:
            c_max_freq, c_max_amp = self.highest_ampl(mask=cur_mask)
            if not config.quiet:
                print(f"Identified peak at f={c_max_freq:.3f} : a={c_max_amp:.3f}")
            c_sig, trough_left_i, trough_right_i = self.get_peak_sig(center_val_freq=c_max_freq, freq_amp=c_max_amp)
            if c_sig > config.cutoff_sig:
                print(f"Accepted frequency - sig of {c_sig:.3f} < {config.cutoff_sig} ({count})")
                return c_max_freq, c_max_amp
            else:
                if not config.quiet:
                    print(f"Rejected frequency - sig of {c_sig:.3f} < {config.cutoff_sig} ({count})")
                count += 1
                for i in range(trough_left_i, trough_right_i):
                    cur_mask[i] = False
        return None, None

    def fit_self_lopoly(self):
        p0 = [0] * config.poly_order
        self.logx = np.log10(self.lsfreq)
        self.logy = np.log10(self.lsamp)
        p, _ = curve_fit(self.polyfunc, self.logx, self.logy, p0)
        self.polyparams_log = p
        self.polyfit = 10 ** self.polyfunc(self.logx, *p)
        print(f"Complete fit of LO poly to log log periodogram")
        print(f"params = {p}")

    def get_polyfit_at_val(self, frequency):
        logfreq = np.log10(frequency)
        logval = self.polyfunc(logfreq, *self.polyparams_log)
        return 10 ** logval

    def plot_polyfit_log(self, show=False, savename=None):
        pl.plot(self.logx, self.logy, color='black', label='data')
        pl.plot(self.logx, self.polyfunc(self.logx, *self.polyparams_log), color='red', label='order 3 poly fit')
        pl.xlabel("Freq [c/d]")
        pl.ylabel(f"log(Amp [{config.target_dtype}])")
        if savename:
            pl.savefig(savename)
        if show:
            pl.show()
        pl.clf()

    def get_sig_polyfit(self, center_val_freq, freq_amp):
        if self.polyparams_log is None:
            self.fit_self_lopoly()
        return freq_amp / self.get_polyfit_at_val(center_val_freq)

    def plot_polyfit_normspace(self, show=False, savename=None):
        pl.plot(self.lsfreq, self.lsamp, color='black', label='data')
        pl.plot(self.lsfreq, self.polyfit, color='red', label='fit')
        pl.legend()
        pl.xlabel("Freq [c/d]")
        pl.ylabel(f"Amp [{config.target_dtype}]")
        if savename:
            pl.savefig(savename)
        if show:
            pl.show()
        pl.clf()

    def plot(self, xlim=(0, 8), vline=None, show=False, savename=None):
        pl.plot(self.lsfreq, self.lsamp, color='black')
        pl.xlabel("Frequency [c/d]")
        pl.ylabel(f"Amplitude [{config.target_dtype}]")
        pl.xlim(*xlim)
        if vline is not None:
            pl.axvline(vline, color='red')
        if show:
            pl.show()
            pl.clf()
        elif savename:
            pl.savefig(savename)
            pl.clf()
