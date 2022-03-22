from Periodogram import Periodogram
import matplotlib.pyplot as pl
import config


class Lightcurve():
    def __init__(self, time, data, err):
        self.time = time
        self.data = data
        self.err = err

        self.periodogram = Periodogram(time, data)

    def plot(self, show=False, color='black', savename = None, model = None):
        pl.plot(self.time, self.data, linestyle='none', marker='.', markersize=1, color=color)
        if model is not None:
            pl.plot(self.time, model, color='red')
        pl.xlabel("Time [HJD - 2450000]")
        pl.ylabel(f"Brightness [{config.target_dtype}]")
        if show:
            pl.show()
        if savename:
            pl.savefig(savename)
            pl.clf()

    def unpack(self):
        return self.time, self.data, self.err

    def measure_N_eff(self):
        sign_change_count = 0
        for i in range(len(self.time)-1):
            if (self.data[i]>0 and self.data[i+1]<0) or (self.data[i]<0 and self.data[i+1] > 0):
                sign_change_count+=1
        return sign_change_count