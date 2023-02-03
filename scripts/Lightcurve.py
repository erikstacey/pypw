from Periodogram import Periodogram
import matplotlib.pyplot as pl
from config import config


class Lightcurve():
    """
    An object for storing light curves data, and their corresponding Periododgram object.
    Attributes:
        np.array time: The time axis of the light curve
        np.array data: the data axis of the light curve
        np.array err: uncertainties on the data in the light curve
        Periodogram periodogram: The periodogram of the stored light curve
    """
    time = None
    data = None
    err = None
    periodogram = None
    def __init__(self, time, data, err):
        self.time = time
        self.data = data
        self.err = err

        self.periodogram = Periodogram(time, data)

    def plot(self, show=False, color='black', savename = None, model = None):
        """
        A diagnostic function for plotting the stored light curve.
        """
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
        """
        Get the time, data, and uncertainty arrays
        :return: time array, data array, uncertainty array
        """
        return self.time, self.data, self.err

    def measure_N_eff(self):
        """
        Measure the number of sign changes in the light curve.
        :return int: number of sign changes
        """
        sign_change_count = 0
        for i in range(len(self.time)-1):
            if (self.data[i]>0 and self.data[i+1]<0) or (self.data[i]<0 and self.data[i+1] > 0):
                sign_change_count+=1
        return sign_change_count