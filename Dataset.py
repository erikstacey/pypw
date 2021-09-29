from Lightcurve import Lightcurve
from Freq import Freq
import numpy as np
import matplotlib.pyplot as pl
import optimization as opt
import config


class Dataset():
    def __init__(self, time, data, err):
        self.time = time
        self.data = data
        self.err = err
        self.LC0 = Lightcurve(time=time, data=data, err=err)

        self.freqs = []
        self.lcs = [self.LC0]

    def it_pw(self):
        print(f"\n\n\t\t ===== ITERATION {len(self.freqs)+1} =====")
        # do single-frequency fit
        print(f"SF Fit:")
        c_lc = self.lcs[-1]
        sf_f_guess, sf_a_guess = c_lc.periodogram.highest_ampl()
        sf_f, sf_a, sf_p, sf_mod = opt.sf_opt_lm(c_lc.time, c_lc.data, c_lc.err, f0=sf_f_guess, a0 = sf_a_guess, p0=0.5)
        self.freqs.append(Freq(sf_f, sf_a, sf_p, len(self.freqs)))


        self.freqs[-1].prettyprint()

        # do multi-frequency fit
        mf_mod = opt.mf_opt_lm(self.LC0.time, self.LC0.data, self.LC0.err, self.freqs)
        print(f"MF Fit:")
        for freq in self.freqs:
            freq.prettyprint()

        # Make residual LC
        if config.residual_model_generation == "sf":
            self.lcs.append(Lightcurve(self.LC0.time, self.lcs[-1].data-sf_mod, self.LC0.err))
        elif config.residual_model_generation == "mf":
            self.lcs.append(Lightcurve(self.LC0.time, self.LC0.data-mf_mod, self.LC0.err))

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