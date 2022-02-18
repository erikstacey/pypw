from Lightcurve import Lightcurve
from Freq import Freq
import numpy as np
import matplotlib.pyplot as pl
import optimization as opt
import config
import os


class Dataset():
    def __init__(self, time, data, err):
        self.time = time
        self.data = data
        self.err = err
        self.LC0 = Lightcurve(time=time, data=data, err=err)

        self.freqs = []
        self.lcs = [self.LC0]

        if config.reg_plots:
            self.LC0.plot(savename=os.getcwd()+config.figure_subdir+"/LC0.png")
            self.LC0.periodogram.plot(savename=os.getcwd() + config.figure_subdir + "/PG0.png")


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
        if config.runtime_plots:
            print("PLOTTING SINGLE FREQUENCY FIT")
            self.lcs[-1].plot(show=False, color='black')
            self.freqs[-1].diag_plot(self.lcs[-1].time, show=True)
            pl.clf()

        # do multi-frequency fit
        mf_mod = opt.mf_opt_lm(self.LC0.time, self.LC0.data, self.LC0.err, self.freqs)
        print(f"MF Fit:")
        for freq in self.freqs:
            freq.prettyprint()

        if config.runtime_plots:
            print("PLOTTING MULTI-FREQUENCY FIT")
            self.lcs[0].plot(show=True, color='black', model = self.total_model())
            pl.clf()


        # save plots
        if config.plot_iterative:
            self.lcs[-1].plot(savename=os.getcwd()+config.iterative_subdir+f"/lc{len(self.lcs)}.png", model=mf_mod)
            self.lcs[-1].periodogram.plot(savename=os.getcwd() + config.iterative_subdir + f"/pg{len(self.lcs)}.png")


        # Make residual LC
        if config.residual_model_generation == "sf":
            self.lcs.append(Lightcurve(self.LC0.time, self.lcs[-1].data-sf_mod, self.LC0.err))
        elif config.residual_model_generation == "mf":
            self.lcs.append(Lightcurve(self.LC0.time, self.LC0.data-mf_mod, self.LC0.err))

        return 0
    def save_results(self, f_name):
        with open(f_name, 'w') as f:
            f.write(f"Freq,Amp,Phase\n")
            for freq in self.freqs:
                f.write(f"{freq.f},{freq.a},{freq.p}\n")

    def save_sf_results(self, f_name):
        with open(f_name, 'w') as f:
            f.write(f"Freq0,Amp0,Phase0\n")
            for freq in self.freqs:
                f.write(f"{freq.f0},{freq.a0},{freq.p0}\n")

    def total_model(self):
        model = np.zeros(len(self.LC0.time))
        for freq in self.freqs:
            model += freq.genmodel(self.LC0.time)
        return model