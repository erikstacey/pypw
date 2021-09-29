from Periodogram import Periodogram
import matplotlib.pyplot as pl


class Lightcurve():
    def __init__(self, time, data, err):
        self.time = time
        self.data = data
        self.err = err

        self.periodogram = Periodogram(time, data)

    def diag_plot(self, show=True, color='black', savename = None):
        pl.plot(self.time, self.data, linestyle='none', marker='.', markersize=1, color=color)
        if show:
            pl.show()
        if savename:
            pl.savefig(savename)
            pl.clf()
    def unpack(self):
        return self.time, self.data, self.err