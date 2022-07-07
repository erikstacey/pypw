from Lightcurve import Lightcurve
from Freq import Freq
import numpy as np
import matplotlib.pyplot as pl
import optimization as opt
import config
import os
from pw_io import format_output
from Periodogram import Periodogram


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
        if config.save_supp_data:
            self.save_data("LC0.csv", self.LC0.time, self.LC0.data, self.LC0.err)
            self.save_data("PG0.csv", self.LC0.periodogram.lsfreq, self.LC0.periodogram.lsamp)


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

        if not config.quiet:
            print(f"Current STD of residuals: {np.std(self.lcs[-1].data)}")

        if config.runtime_plots:
            print("PLOTTING MULTI-FREQUENCY FIT")
            self.lcs[0].plot(show=True, color='black', model = self.total_model())
            pl.clf()


        # save plots
        if config.plot_iterative:
            self.lcs[0].plot(savename=os.getcwd()+config.iterative_subdir+f"/model_lc{len(self.lcs)}.png", model=mf_mod)
            self.lcs[-1].plot(savename=os.getcwd() + config.iterative_subdir + f"/lc{len(self.lcs)}.png")
            self.lcs[-1].periodogram.plot(savename=os.getcwd() + config.iterative_subdir + f"/pg{len(self.lcs)}.png")
            if config.peak_selection == "slf" and self.lcs[-1].periodogram.slf_p is not None:
                self.lcs[-1].periodogram.plot_slf_noise(show=False, savename=os.getcwd() + config.iterative_subdir + f"/pg_slf_fit_{len(self.lcs)}.png")


        # Make residual LC
        if config.residual_model_generation == "sf":
            self.lcs.append(Lightcurve(self.LC0.time, self.lcs[-1].data-sf_mod, self.LC0.err))
        elif config.residual_model_generation == "mf":
            self.lcs.append(Lightcurve(self.LC0.time, self.LC0.data-mf_mod, self.LC0.err))

        # save LC and periodograms if specified
        if config.save_supp_data:
            self.save_data(f"LC{len(self.freqs)}.csv", self.lcs[-1].time, self.lcs[-1].data, self.lcs[-1].err)
            self.save_data(f"PG{len(self.freqs)}.csv", self.lcs[-1].periodogram.lsfreq, self.lcs[-1].periodogram.lsamp)



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
        if config.reg_plots:
            self.lcs[-1].periodogram.plot_polyfit_log(show = False,
                                                      savename = f"{config.working_dir}/{config.figure_subdir}/polyfig_log")
            self.lcs[-1].periodogram.plot_polyfit_normspace(show = False,
                                                            savename = f"{config.working_dir}/{config.figure_subdir}/polyfig_normspace")
            self.lcs[-1].periodogram.plot_slf_noise(show=False,
                                                    savename=f"{config.working_dir}/{config.figure_subdir}/slf_fit")
            self.lcs[-1].periodogram.plot_slf_noise(show=False,
                                                    savename=f"{config.working_dir}/{config.figure_subdir}/slf_fit_logy",
                                                    logy = True)



    def save_results(self, f_name):
        with open(f_name, 'w') as f:
            f.write(f"Freq,Amp,Phase, Freq_err, amp_err, phase_err, sig_poly, sig_avg, sig_slf\n")
            for freq in self.freqs:
                f.write(f"{freq.f},{freq.a},{freq.p}, {freq.f_err}, {freq.a_err}, {freq.p_err}, {freq.sig_poly}, {freq.sig_avg}, {freq.sig_slf}\n")

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

    def save_data(self, filename, x, y, err=None):
        with open(f"supp_data/{filename}", 'w') as f:
            for i in range(len(x)):
                if err is not None:
                    f.write(f"{x[i]},{y[i]},{err[i]}\n")
                else:
                    f.write(f"{x[i]},{y[i]}\n")

    def save_results_latex(self, filename):
        with open(filename, 'w') as f:
            f.write("\\begin{table}[] \n \\begin{tabular}{llllll} \n Index & Frequency {[}c/d{]} & Amplitude {[}mmag{]} & Phase      & Significance & Notes \\\\ \n")
            c = 1
            for freq in self.freqs:
                f.write(f"{c} & {format_output(freq.f, freq.f_err, 2)} & "
                        f"{format_output(freq.a, freq.a_err, 2)} & "
                        f"{format_output(freq.p, freq.p_err, 2)} &"
                        f"{round(freq.sig_avg, 2)} / {round(freq.sig_poly, 2)} & \\\\ \n")
                c+=1
            f.write("\\end{tabular} \n \\end{table}")

    def save_misc(self, filename):
        with open(filename, 'w') as f:
            f.write(f"r_polyparams_log={self.lcs[-1].periodogram.polyparams_log}\n")
            f.write(f"r_slfparams={self.lcs[-1].periodogram.slf_p}\n")

            stddevs = [np.std(lc.data) for lc in self.lcs]
            f.write(f"std_devs={stddevs}\n")