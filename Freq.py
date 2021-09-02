import numpy as np
from models import sin_model
import matplotlib.pyplot as pl

class Freq():
    def __init__(self, freq, amp, phi, n=None):
        self.freq = freq
        self.amp = amp
        self.phi = phi
        if n is not None:
            self.n = n
        self.sig = None
        self.freq_err = None
        self.amp_err = None
        self.phase_err = None

        self.f0 = freq
        self.a0 = amp
        self.p0 = phi
        self.adjust_params()
    def update(self, newfreq,newamp, newphase):
        self.freq, self.amp, self.phi = newfreq, newamp, newphase
    def triplet_list(self):
        return [self.freq, self.amp, self.phi]
    def prettyprint(self):
        print(f"\t{self.n} // f = {self.freq:.5f} | a = {self.amp:.3f} | phi = {self.phi:.3f}")

    def prettyprint_0(self):
        print(f"{self.n}(original) // f = {self.f0:.5f} | a = {self.a0:.3f} | phi = {self.p0:.3f}")

    def prettyprint_sig(self):
        print(f"{self.n} // f = {self.freq:.5f} | a = {self.amp:.3f} | phi = {self.phi:.3f} ({self.sig})")
    def genmodel(self, time):
        return sin_model(time, self.freq, self.amp, self.phi)

    def adjust_params(self):
        pass
        # adjust amplitude
        if self.amp < 0:
            self.amp = abs(self.amp)
            self.phi+=0.5

        if self.phi > 1 or self.phi < 0:
            self.phi = self.phi % 1

        self.f0_adjusted, self.a0_adjusted, self.p0_adjusted = self.freq, self.amp, self.phi
    def assign_errors(self, rho, N, T, sigmaresiduals):
        self.freq_err = (6/N)**0.5 * sigmaresiduals / (np.pi*T*self.amp)
        self.amp_err = (2/N)**0.5 * sigmaresiduals
        self.phase_err = (2/N)**0.5 *sigmaresiduals / self.amp
    def diag_plot(self, x, show=True, color='red'):
        pl.plot(x, self.genmodel(x), color=color)
        if show:
            pl.show()