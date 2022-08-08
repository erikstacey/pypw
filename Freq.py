import numpy as np
from models import sin_model
import matplotlib.pyplot as pl


class Freq():
    def __init__(self, freq, amp, phi, n):
        self.f = freq
        self.a = amp
        self.p = phi
        self.n = n
        self.sig_poly = None
        self.sig_avg = None
        self.sig_slf = None
        self.f_err = None
        self.a_err = None
        self.p_err = None

        self.f0 = freq
        self.a0 = amp
        self.p0 = phi
        self.adjust_params()

    def update(self, newfreq, newamp, newphase):
        self.f, self.a, self.p = newfreq, newamp, newphase

    def triplet_list(self):
        return [self.f, self.a, self.p]

    def prettyprint(self):
        print(f"\t{self.n} // f = {self.f:.5f} | a = {self.a:.3f} | phi = {self.p:.3f}")

    def prettyprint_0(self):
        print(f"{self.n}(original) // f = {self.f0:.5f} | a = {self.a0:.3f} | phi = {self.p0:.3f}")

    def prettyprint_sig(self):
        print(f"{self.n} // f = {self.f:.5f} | a = {self.a:.3f} | phi = {self.p:.3f} ({self.sig})")

    def genmodel(self, time):
        return sin_model(time, self.f, self.a, self.p)

    def genmodel_sf(self, time):
        return sin_model(time, self.f0, self.a0, self.p0)

    def adjust_params(self):
        pass
        # adjust amplitude
        if self.a < 0:
            self.a = abs(self.a)
            self.p += 0.5

        if self.p > 1 or self.p < 0:
            self.p = self.p % 1

        self.f0_adjusted, self.a0_adjusted, self.p0_adjusted = self.f, self.a, self.p

    def assign_errors(self, N, T, sigmaresiduals):
        self.f_err = (6 / N) ** 0.5 * sigmaresiduals / (np.pi * T * self.a)
        self.a_err = (2 / N) ** 0.5 * sigmaresiduals
        self.p_err = (2 / N) ** 0.5 * sigmaresiduals / self.a
