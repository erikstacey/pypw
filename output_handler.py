import config
import os
import shutil
import numpy as np

import matplotlib.pyplot as pl
from models import flux2mag, flux2mag_e


class OutputHandler():
    def __init__(self, LC0, reference_flux=None):
        self.rflux = reference_flux

        self.model_x = np.linspace(LC0.time[0], LC0.time[-1], 10000)

        # remove any existing results and (re-)establish the tree
        self.main_dir = f"{config.working_dir}/{config.main_output}"
        if os.path.isdir(self.main_dir):
            shutil.rmtree(f"{config.working_dir}/{config.main_output}")

        os.mkdir(self.main_dir)

        # set up light curves directories and subdirectories
        self.pgs_output = f"{self.main_dir}/{config.pgs_output}"
        self.pgs_slf_output = f"{self.pgs_output}/slf_fits"
        self.pgs_box_avg_output = f"{self.pgs_output}/box_avg"
        self.pgs_lopoly_output = f"{self.pgs_output}/lopoly"
        self.pgs_data_output = f"{self.pgs_output}/data"
        os.mkdir(self.pgs_output)
        os.mkdir(self.pgs_slf_output)
        os.mkdir(self.pgs_box_avg_output)
        os.mkdir(self.pgs_lopoly_output)
        os.mkdir(self.pgs_data_output)

        self.lcs_output = f"{self.main_dir}/{config.lcs_output}"
        self.lcs_sf_output = f"{self.lcs_output}/sf_fits"
        self.lcs_mf_output = f"{self.lcs_output}/mf_fits"
        self.lcs_data_output = f"{self.lcs_output}/data"

        os.mkdir(self.lcs_output)
        os.mkdir(self.lcs_sf_output)
        os.mkdir(self.lcs_mf_output)
        os.mkdir(self.lcs_data_output)

        self.misc_output = f"{self.main_dir}/{config.misc_output}"
        os.mkdir(self.misc_output)

        self.save_lc(LC0, f"{self.main_dir}/{config.preprocessed_lightcurve_fname}")



    def save_it(self, lcs, freqs):
        n = len(freqs) - 1
        c_lc = lcs[-1]
        c_pg = c_lc.periodogram
        c_freq = freqs[-1]

        # Save LC, PG data
        self.save_lc(c_lc, f"{self.lcs_data_output}/lc{n}.csv")
        self.save_pg(c_pg, f"{self.pgs_data_output}/pg{n}.csv")

        # ================= LC PLOTS ===================
        # regular ======================================
        self.plot_lc(x=c_lc.time, y=c_lc.data)
        self.format_lc()
        pl.savefig(f"{self.lcs_output}/lc{n}.png")
        pl.clf()
        # sf ===========================================
        self.plot_lc(x=c_lc.time, y=c_lc.data, label="Data")
        self.plot_lc(x=self.model_x, y=c_freq.genmodel_sf(self.model_x), color="red", label="SF Model")
        self.format_lc(legend=True)
        pl.savefig(f"{self.lcs_sf_output}/lc_sf{n}.png")
        pl.clf()
        # mf ===========================================
        self.plot_lc(x=lcs[0].time, y=lcs[0].data, label="Data")
        mf_y = np.zeros(len(self.model_x))
        for freq in freqs:
            mf_y += freq.genmodel(self.model_x)
        self.plot_lc(x=self.model_x, y=mf_y, color="red",label="MF Model")
        self.format_lc(legend=True)
        pl.savefig(f"{self.lcs_mf_output}/lc_mf{n}.png")
        pl.clf()

        # ================= LC PLOTS ===================
        # regular ======================================
        self.plot_pg(x=c_pg.lsfreq, y=c_pg.lsamp, label="Data")
        self.format_pg()
        pl.savefig(f"{self.pgs_output}/pg{n}.png")
        pl.clf()
        # slf ==========================================
        if c_pg.slf_p is not None:
            self.plot_pg(x=c_pg.lsfreq, y=c_pg.lsamp, label="Data")
            self.plot_pg(x=c_pg.lsfreq, y=c_pg.get_slf_model(c_pg.lsfreq), color="red", label="SLF Fit")
            self.format_pg()
            pl.savefig(f"{self.pgs_slf_output}/pg{n}.png")
            pl.clf()
        # lopoly =======================================
        if c_pg.polyparams_log is not None:
            self.plot_pg(x=c_pg.lsfreq, y=c_pg.lsamp, label="Data")
            self.plot_pg(x=c_pg.lsfreq, y=c_pg.get_polyfit_model(c_pg.lsfreq), color="red", label="LOPoly Fit")
            self.format_pg()
            pl.savefig(f"{self.pgs_lopoly_output}/pg{n}.png")
            pl.clf()

    def post_pw(self, lcs, freqs):
        self.save_freqs(freqs, f"{self.main_dir}/{config.frequencies_fname}")

    def plot_lc(self, x, y, color="black", marker=".", alpha=1.0, linestyle="none", label="", markersize = 1):
        if config.output_in_mmag:
            pl.plot(x, flux2mag(y, self.rflux)*1000,
                    color=color, marker=marker, alpha=alpha, linestyle=linestyle, label=label, markersize=markersize)
        else:
            pl.plot(x, y,
                    color=color, marker=marker, alpha=alpha, linestyle=linestyle, label=label, markersize=markersize)

    def format_lc(self, legend=False):
        pl.xlabel(config.lc_xlabel)
        pl.ylabel(config.lc_ylabel)
        if legend:
            pl.legend()

    def plot_pg(self, x, y, color="black", alpha=1.0, label=""):
        if config.output_in_mmag:
            pl.plot(x, -flux2mag(y, self.rflux)*1000,
                    color=color, alpha=alpha, label=label)
        else:
            pl.plot(x, y,
                    color=color, alpha=alpha, label=label)

    def format_pg(self, legend=False):
        pl.xlabel(config.pg_xlabel)
        pl.ylabel(config.pg_ylabel)
        if legend:
            pl.legend()


    def get_freq_params_in_mmag(self, freq):
        a, a_err, p, a0, p0 = freq.a, freq.a_err, freq.p, freq.a0, freq.p0
        am, a_errm = flux2mag_e(a, self.rflux, a_err)
        a0m = flux2mag(a0, self.rflux)
        # flip so it's positive
        am *=-1
        # adjust phase appropriately
        pm = p - 0.5
        p0m = p0-0.5
        if pm < 0:
            pm += 1
        if p0m < 0:
            p0m+=1

        return am*1000, a_errm*1000, pm, a0m*1000, p0m*1000

    def save_freqs(self, freqs, path):
        with open(path, "w") as f:
            f.write(f"Freq,Amp,Phase,Freq_err,Amp_err,Sig_slf,Sig_lopoly,Sig_avg")
            if config.output_in_mmag:
                for freq in freqs:
                    am, a_errm, pm, _, _ = self.get_freq_params_in_mmag(freq)
                    f.write(f"{freq.f},{am},{pm},{freq.f_err},{a_errm},{freq.p_err},{freq.sig_slf},{freq.sig_poly},{freq.sig_avg}")
            else:
                for freq in freqs:
                    f.write(f"{freq.f},{freq.a},{freq.m},{freq.f_err},{freq.a_err},{freq.p_err},{freq.sig_slf},{freq.sig_poly},{freq.sig_avg}")



    def save_lc(self, lightcurve, path):
        np.savetxt(path, X=[lightcurve.time, lightcurve.data, lightcurve.err], delimiter=",")

    def save_pg(self, periodogram, path):
        np.savetxt(path, X=[periodogram.lsfreq, periodogram.lsamp], delimiter=",")

