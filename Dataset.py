from Lightcurve import Lightcurve
from Freq import Freq
import numpy as np
import matplotlib.pyplot as pl
import optimization as opt
import config
import os
from pw_io import format_output
from Periodogram import Periodogram
from output_handler import OutputHandler


class Dataset():
    def __init__(self, time, data, err, reference_flux = None):
        self.time = time
        self.data = data
        self.err = err
        self.LC0 = Lightcurve(time=time, data=data, err=err)

        self.freqs = []
        self.lcs = [self.LC0]

        self.output_handler = OutputHandler(LC0 = self.LC0, reference_flux = reference_flux)




    def it_pw(self):
        print(f"\n\n\t\t ===== ITERATION {len(self.freqs)+1} =====")
        # do single-frequency fit
        print(f"SF Fit:")
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
                sf_f_guess, sf_a_guess = c_lc.periodogram.peak_selection_slf_fits()
        if sf_f_guess == None:
            print("STOP CRITERION TRIGGERED")
            return 1
        sf_p_guess = 0.5
        while True:
            sf_f, sf_a, sf_p, sf_mod = opt.sf_opt_lm(c_lc.time, c_lc.data, c_lc.err, f0=sf_f_guess, a0 = sf_a_guess, p0=sf_p_guess)
            # check if phase has moved or not
            if abs(sf_p-sf_p_guess) > 0.1:
                print("Phase fit check fine")
                break
            else:
                print(f"SF Fit of phase {sf_p:.3f} too close to guess of {sf_p_guess:.3f}; adjusting by 0.11")
                sf_p_guess += 0.11
        self.freqs.append(Freq(sf_f, sf_a, sf_p, len(self.freqs)))
        self.freqs[-1].prettyprint()
        # do multi-frequency fit
        mf_mod = opt.mf_opt_lm(self.LC0.time, self.LC0.data, self.LC0.err, self.freqs)
        print(f"MF Fit:")
        for freq in self.freqs:
            freq.prettyprint()

        self.output_handler.save_it(self.lcs, self.freqs)
        if not config.quiet:
            print(f"Current STD of residuals: {np.std(self.lcs[-1].data)}")
        # Make residual LC
        if config.residual_model_generation == "sf":
            self.lcs.append(Lightcurve(self.LC0.time, self.lcs[-1].data-sf_mod, self.LC0.err))
        elif config.residual_model_generation == "mf":
            self.lcs.append(Lightcurve(self.LC0.time, self.LC0.data-mf_mod, self.LC0.err))




        return 0
    def post(self):
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
        self.output_handler.post_pw(self.lcs, self.freqs)

    def total_model(self):
        model = np.zeros(len(self.LC0.time))
        for freq in self.freqs:
            model += freq.genmodel(self.LC0.time)
        return model
