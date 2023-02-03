import numpy as np
from scipy.optimize import curve_fit
from models import n_model_poly, bowman_noise_model
from astropy.timeseries import LombScargle
from config import config


class Periodogram:
    """
    Stores a periodogram and the parameters of fits to the periodogram. Includes functionality to perform fits to the
    periodogram and measure significances at given values of frequency.

    Attributes:
        np.array lsfreq: An array of periodogram frequency values
        np.array lsamp: An array of periodogram amplitude values with the same units as the input time series
        float resolution: The minimum frequency separation required to properly resolve two candidate periodic components
        func polyfunc: The function of the format polyfunc(x, *p) used for the low-order polynomial fit
        np.array polyfit: The polynomial fit evaluated at the x values of the periodogram
        list polyparams_log: The parameters of a polynomial fit in log-log space.
        np.array logx: The frequencies of the periodogram in log base 10
        np.array logy: The amplitudes of the periodogram in log base 10
        list slf_p: The parameters of a SLF variability fit using the Bowman et al. (2019) model
        list slf_p_err: The least-squares uncertainties on the parameters stored in slf_p
    """
    lsfreq=None
    lsamp=None
    resolution=None
    polyfunc=None
    polyfit = None
    polyparams_log = None
    logx = None
    logy = None
    slf_p = None
    slf_p_err = None

    def __init__(self, time, data):
        """
        Initialize a new periodogram object from a time series. Compute a new periodogram and store in lsfreq and lsamp
        attributes
        :param np.array time: An array of time stamps for a time series
        :param np.array data: An array of measurements for a time series
        """
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




    def highest_ampl(self, mask=None):
        """
        Return the frequency and amplitude pair corresponding to the highest amplitude in the periodogram.
        :param list mask: an optional boolean exclusion mask of the same length as the periodogram axes.
        :return float freq, float amp: the frequency and amplitude of the highest peak on the periodogram
        """
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
        """
        Determine local minima in the periodogram on either side of a specified value
        :param float center: the central specified value
        :return int left_i, int right_i: the indices corresponding to local minima on the left and right sides of the
        specified value
        """
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
            if right_i > len(self.lsfreq):
                left_i = len(self.lsfreq)
        return left_i, right_i

    def find_index_of_freq(self, t):
        """
        Find the index of the closest frequency in the frequency axis of the periodogram
        :param float t: the target frequency
        :return int: The index of the closest frequency in the periodogram frequency axis to the target value t
        """
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
        """
        Find the significance of a frequency with a specified amplitude using the box average method with consideration
        for the peak width (e.g. finds the local minima on either side of the freq value, then averages in a box around
        those minima)
        :param float center_val_freq: The specified frequency
        :param float freq_amp: The amplitude of the specified frequency
        :return float, int, int: The measured significance, the index of the left local minimum around the peak,
        the index of the right local minimum around the peak
        """
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
                f"\tAveraged from {lower_val_freq:.3f}:{self.lsfreq[trough_left_i]:.3f} and "
                f"{self.lsfreq[trough_right_i]:.3f}:{upper_val_freq:.3f}")
            print(f"\tYielded avg = {avg_regions_avg:.3f} ||| Nom. Freq. amp = {self.lsamp[center_i_freq]:.3f}")
        return freq_amp / avg_regions_avg, trough_left_i, trough_right_i

    def get_sig_boxavg(self, center_val_freq, freq_amp):
        """
        Find the significance of a frequency with a specified amplitude using the box average method without
        consideration for the peak width.
        :param float center_val_freq: The specified frequency
        :param float freq_amp: The amplitude of the specified frequency
        :return float: The measured significance
        """
        center_i_freq = self.find_index_of_freq(center_val_freq)
        # find values of edge frequencies
        lower_val_freq = self.lsfreq[center_i_freq] - config.averaging_bin_radius
        upper_val_freq = self.lsfreq[center_i_freq] + config.averaging_bin_radius
        # find left and right indices
        lower_i_freq = self.find_index_of_freq(lower_val_freq)
        upper_i_freq = self.find_index_of_freq(upper_val_freq)
        total_avg_region = self.lsamp[lower_i_freq:upper_i_freq]
        return freq_amp / np.mean(total_avg_region)

    def peak_selection_w_sig(self):
        """
        Find a frequency/amplitude candidate pair in the periodogram, requiring that the peak exceed the cutoff
        significance specified in the config according to a box average with consideration for the peak width
        :return float x, float y: Returns a candidate frequency x and its corresponding amplitude y. If no peaks
        exceed the cutoff significance, return None, None.
        """
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

    def peak_selection_slf_fits_old(self):
        """Deprecated, but kept here for testing purposes"""
        # set up slf noise fit
        if self.slf_p is None:
            self.fit_self_slfnoise()
        cur_mask = np.ones(len(self.lsfreq), dtype=bool)
        count = 0
        while count < config.cutoff_iteration:
            c_max_freq, c_max_amp = self.highest_ampl(mask=cur_mask)
            if not config.quiet:
                print(f"Identified peak at f={c_max_freq:.3f} : a={c_max_amp:.3f}")
            c_sig = c_max_amp/bowman_noise_model(c_max_freq, *self.slf_p)
            if c_sig > config.cutoff_sig:
                print(f"Accepted frequency - sig of {c_sig:.3f} < {config.cutoff_sig} ({count})")
                return c_max_freq, c_max_amp
            else:
                print(f"Rejected frequency - sig of {c_sig:.3f} < {config.cutoff_sig} ({count})")
                trough_left_i, trough_right_i = self.find_troughs(self.find_index_of_freq(c_max_freq))
                cur_mask[trough_left_i:trough_right_i] = False
                count+=1
        return None, None

    def peak_selection_slf_fits(self, freqs):
        """
        Find a frequency/amplitude candidate pair requiring that the amplitude exceed a provisional SLF fit by a
        config-specified significance criterion (cutoff_sig)
        :param list freqs: A list of previously-identified frequencies
        :return float f, float a: The identified frequency (f) and amplitude (a) pair. Returns None, None if no peak is
        found.
        """
        freq_values = np.array([freq.f for freq in freqs])
        # set up slf noise fit
        if self.slf_p is None:
            self.fit_self_slfnoise()
        max_a = 0
        cor_f = 0
        # iterate over periodogram and check for significance and nearby previously-identified frequencies
        for i in range(len(self.lsfreq)):
            if self.lsamp[i] > max_a and self.lsamp[i] > \
                    config.cutoff_sig * bowman_noise_model(self.lsfreq[i], *self.slf_p):
                if config.prevent_frequencies_within_resolution and \
                        np.all(abs(freq_values - self.lsfreq[i]) > self.resolution):
                    max_a = self.lsamp[i]
                    cor_f = self.lsfreq[i]
                else:
                    max_a = self.lsamp[i]
                    cor_f = self.lsfreq[i]

        if max_a == 0:
            print("Frequency not found - Terminating analysis")
            return None, None
        else:
            print(f"Highest frequency above the SLF fit is {cor_f} "
                  f"(A={max_a:.2f}, sig = {max_a/bowman_noise_model(cor_f, *self.slf_p):.2f})")
            return cor_f, max_a

    def fit_self_lopoly(self):
        """
        Fit the log-log periodogram with a low-order polynomial of degree specified by config.poly_order and store the
        parameters in the attribute polyparams_log
        :return: None
        """
        p0 = [0] * config.poly_order
        self.logx = np.log10(self.lsfreq)
        self.logy = np.log10(self.lsamp)
        p, _ = curve_fit(self.polyfunc, self.logx, self.logy, p0)
        self.polyparams_log = p
        self.polyfit = 10 ** self.polyfunc(self.logx, *p)
        print(f"Complete fit of LO poly to log log periodogram")
        print(f"params = {p}")

    def fit_self_slfnoise(self):
        """
        Fit the periodogram with a slf noise model as specified in Bowman et al. (2019) and store the parameters in
        attributes slf_p for the parameters and slf_p_err for the least-squares parameter uncertainties
        :return: None
        """
        p0 = [0.5, np.mean(self.lsamp), 0.5, 0]

        p, covar = curve_fit(bowman_noise_model,xdata=self.lsfreq, ydata=self.lsamp, p0=p0)
        self.slf_p = p
        self.slf_p_err = np.array([covar[i,i] for i in range(len(p))])
        print(f"Completed red noise + white noise model fit ")
        print(f"params = {p}")

    def get_sig_slf(self, f, a):
        """
        Find the significance of a specified frequency/amplitude pair by comparison with a slf model fit
        :param float f: A specified frequency
        :param float a: The measured amplitude at the frequency f
        :return float: The measured significance
        """
        if self.slf_p is None:
            self.fit_self_slfnoise()
        model_at_val = bowman_noise_model(f, *self.slf_p)
        return a/model_at_val

    def get_slf_model(self, x):
        """
        Find the value of the currently-stored SLF model fit at the specified frequency x
        :param float x: a specified frequency
        :return float: The value of the SLF variability model at the frequency x
        """
        return bowman_noise_model(x, *self.slf_p)

    def get_polyfit_model(self, x):
        """
        Find the value of the currently-stored low-order polynomial model fit at the specified frequency x
        :param float x: a specified frequency
        :return float: The value of the LOPoly model at the frequency x
        """
        return 10**self.polyfunc(np.log10(x), *self.polyparams_log)

    def get_polyfit_model_log(self, x):
        """
        Find the value of the currently-stored low-order polynomial model fit at the specified frequency x in log space
        :param float x: a specified frequency
        :return float, float: The log10 value of x and the value of the LOPoly model at the frequency x
        """
        return np.log10(x), self.polyfunc(np.log10(x), *self.polyparams_log)

    def get_polyfit_at_val(self, frequency):
        """
        Does the same thing as get_polyfit_model. %todo: remove one or both of get_polyfit_model and get_polyfit_at_val
        """
        logfreq = np.log10(frequency)
        logval = self.polyfunc(logfreq, *self.polyparams_log)
        return 10 ** logval



    def get_sig_polyfit(self, center_val_freq, freq_amp):
        """
        Find the significance of a frequency with a measured amplitude according to a log-log space polynomial fit
        :param float center_val_freq: A specified frequency
        :param float freq_amp: A measured amplitude
        :return float: The determined significance
        """
        if self.polyparams_log is None:
            self.fit_self_lopoly()
        return freq_amp / self.get_polyfit_at_val(center_val_freq)
