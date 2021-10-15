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
        pl.xlabel("Time [HJD]")
        pl.ylabel(f"Brightness [{config.target_dtype}]")
        if show:
            pl.show()
        if savename:
            pl.savefig(savename)
            pl.clf()

    def unpack(self):
        return self.time, self.data, self.err