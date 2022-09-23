from Lightcurve import Lightcurve
from Freq import Freq
import numpy as np
import optimization as opt
from config import config
from output_handler import OutputHandler


class Dataset:
    """
    A class to store Lightcurve/Frequency objects pertaining to a pre-whitening analysis of a single time series.
    Methods of this class mechanically manage the pre-whitening as called from the main function.
    Attributes:
        np.array time: The time axis of the input time series
        np.array data: The data axis of the input time series
        np.array err: Uncertainties on the input data. If no uncertainties exist for the data this should be set to an
            array of ones.
        Lightcurve LC0: A Lightcurve object storing the input time series
        list freqs: A list of Freq objects storing the results of the pre-whitening analysis. A new freq is added
            each time it_pw is called, except when the analysis is terminated
        list lcs: A list of Lightcurve objects storing the original (at index 0) and residual light curves from the
            pre-whitening analysis
        OutputHandler output_handler: For outputting plots and auxiliary data
        float c_zp: a floating mean adjustment. If config.multi_fit_zp is True, this stores the zero-point offset param
            from the complete variability model optimizations
    """

    time = None
    data = None
    err = None
    LC0 = None
    freqs = None
    lcs = None
    output_handler = None
    c_zp = None
    def __init__(self, time, data, err, reference_flux = None):
        """
        Contructor for the Dataset class
        :param np.array time: The time axis of the time series
        :param np.array data: The data axis of the time series
        :param np.array err: The uncertainties of the time series.
        :param float reference_flux: An optional reference flux value used for conversions to differential magnitude
            passed to the output_handler
        """
        self.time = time
        self.data = data
        self.err = err
        self.LC0 = Lightcurve(time=time, data=data, err=err)
        self.freqs = []
        self.lcs = [self.LC0]
        self.output_handler = OutputHandler(LC0 = self.LC0, reference_flux = reference_flux)
        self.c_zp = 0
        print(f"Dataset loaded with n={len(time)} and deltaT = {time[-1]-time[0]}")

    def it_pw(self):
        """
        Perform a pre-whitening iteration. This appends a new frequency to self.freqs and updates the parameters of
        all frequencies contained therein. This also appends a light curve to self.lcs.
        :return: 0 if iteration was successful, 1 if no peak could be found during peak selection
        """
        print(f"\n\n\t\t ===== ITERATION {len(self.freqs)+1} =====")
        # do single-frequency fit
        print("Identifying Frequency...")
        c_lc = self.lcs[-1]
        if config.peak_selection == "highest":
            sf_f_guess, sf_a_guess = c_lc.periodogram.highest_ampl()
        elif config.peak_selection == "bin":
            if len(self.freqs)+1 <= config.bin_highest_override:
                sf_f_guess, sf_a_guess = c_lc.periodogram.highest_ampl()
            else:
                sf_f_guess, sf_a_guess = c_lc.periodogram.peak_selection_w_sig()
        elif config.peak_selection == "slf":
            if len(self.freqs)+1 <= config.bin_highest_override:
                sf_f_guess, sf_a_guess = c_lc.periodogram.highest_ampl()
            else:
                sf_f_guess, sf_a_guess = c_lc.periodogram.peak_selection_slf_fits(self.freqs)
        if sf_f_guess == None:
            print("STOP CRITERION TRIGGERED")
            return 1
        print(f"SF Fit:")
        sf_p_guess = 0.5
        while True:
            sf_f, sf_a, sf_p, sf_mod = opt.sf_opt_lm(c_lc.time, c_lc.data, c_lc.err, f0=sf_f_guess, a0 = sf_a_guess, p0=sf_p_guess)
            # check if phase has moved or not
            if abs(sf_p-sf_p_guess) > 0.1:
                print("Phase fit check fine")
                break
            else:
                # adjust the phase and try again
                print(f"SF Fit of phase {sf_p:.3f} too close to guess of {sf_p_guess:.3f}; adjusting by 0.11")
                sf_p_guess += 0.11
        self.freqs.append(Freq(sf_f, sf_a, sf_p, len(self.freqs)))
        self.freqs[-1].prettyprint()
        # do multi-frequency fit
        print("MF Free Frequency:")
        mf_mod, self.c_zp = opt.mf_opt_lm(self.LC0.time, self.LC0.data, self.LC0.err, self.freqs, self.c_zp)
        print(f"MF Fit:")
        for freq in self.freqs:
            freq.prettyprint()

        self.output_handler.save_it(self.lcs, self.freqs, self.c_zp)
        if not config.quiet:
            print(f"Current STD of residuals: {np.std(self.lcs[-1].data)}")
        # Make residual LC
        if config.residual_model_generation == "sf":
            self.lcs.append(Lightcurve(self.LC0.time, self.lcs[-1].data-sf_mod, self.LC0.err))
        elif config.residual_model_generation == "mf":
            self.lcs.append(Lightcurve(self.LC0.time, self.LC0.data-mf_mod, self.LC0.err))
        return 0
    def post(self):
        """
        Adjust phases to the interval (0, 1), assign significances, measure formal uncertainties according to
        Montgomery & Odonogue (1999) with the Schwarzenberg-Czerny (2003)/Degroote et al. (2007) adjustments.
        :return: None
        """
        print(f"Final ZP: {self.c_zp}")
        for i in range(len(self.freqs)):
            # adjust parameters
            self.freqs[i].adjust_params()
            # get significances
            self.freqs[i].sig_poly = self.lcs[-1].periodogram.get_sig_polyfit(self.freqs[i].f, self.freqs[i].a)
            self.freqs[i].sig_avg = self.lcs[-1].periodogram.get_sig_boxavg(self.freqs[i].f, self.freqs[i].a)
            self.freqs[i].sig_slf = self.lcs[-1].periodogram.get_sig_slf(self.freqs[i].f, self.freqs[i].a)
            # assign formal errors
            N_eff = self.lcs[-1].measure_N_eff()
            self.freqs[i].assign_errors(N_eff,
                                        self.LC0.time[-1]-self.LC0.time[0],
                                        np.std(self.lcs[-1].data))
        self.output_handler.post_pw(self.lcs, self.freqs, self.c_zp)

    def total_model(self):
        """
        Get the complete variability model evaluated at the times of the input light curve
        :return np.array: The complete variability model
        """
        model = np.zeros(len(self.LC0.time)) + self.c_zp
        for freq in self.freqs:
            model += freq.genmodel(self.LC0.time)
        return model
