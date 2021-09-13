import os
"""
Instructions for use:
1) Set everything under main - recommend making a directory and putting the timeseries file in it, then setting
the directory and target file appropriately.
2) If the data file is processed already, just set the data types to match (dtype and target_dtype), and set strip_gaps
to False.
3) Set other options - some are self-explanatory and others are less. All are poorly documented at the moment.
4) Run main.py
"""
# ========== Main/Most relevant ==========
version = "1.1"
working_dir = f"C:/Users/Erik/main/projects/rho_pup/sec34"
target_file = "rho_pup_lc_sec34.txt"
cols = [0, 1, 2]
dtype = "flux"
n_f = 2  # number of frequencies to extract

averaging_bin_width = 2
cutoff_iteration = 6
cutoff_sig = 3

multi_fit_type = "lm"  # "anneal", "lm", "scipy". Sets fitting engine to use.
residual_model_generation = "mf" # can be sf, mf. Controls which model is used to generate residual periodogram

clean_existing = True

# ========== Preprocessing ==========
target_dtype = "mmag"  # allowed values: mag, mmag, flux (only if data originally in flux)
strip_gaps = True
gap_size = 0.5  # days
strip_gaps_points = 50  # on either side of each gap, and at the start and end
# ========== Logging ==========
log_fname = "log.txt"
save_freq_logs = True
freqlog_folder = "freq_logs"

# ========== Console ==========
quiet = False

# ========== Output targets ==========
preprocessed_lightcurve_fname = "inp_lc.dat"
frequencies_fname = "frequencies.csv"

# ========== Frequency Selection ==========
# TODO: implement LOPOLY fit for freq selection+rejection by significance
freq_selection_method = "highest" # can be highest, or averaged


# ========== Fitting ==========
fixed_freqamp_single = False
fixed_freq_single = False
fixed_none_single = True

phase_fit_rejection_criterion = 0.1

fixed_freqamp_multi = False
fixed_freq_multi = True
fixed_none_multi = True
# ========== LM ==========
# freq bounds use coefficients
lm_freq_bounds = True
lm_freq_bounds_lower_coef = 0.8
lm_freq_bounds_upper_coef = 1.2
# amp bounds use coefficients
lm_amp_bounds = True
lm_amp_bounds_lower_coef = 0.8
lm_amp_bounds_upper_coef = 1.2
# phase bounds set explicitly
lm_phase_bounds = True
lm_phase_bounds_lower = -100
lm_phase_bounds_upper = 100
# ========== Annealing ==========
ann_bounds_method = "relative"  # "explicit", "relative", "plusminus"
if ann_bounds_method=="explicit":
    ann_freq_bounds_lower = 0
    ann_freq_bounds_upper = 20
    ann_amp_bounds_lower = 0
    ann_amp_bounds_upper = 20
    ann_phase_bounds_lower = 0
    ann_phase_bounds_upper = 1
if ann_bounds_method == "relative":
    ann_relfrac_freq = 0.1
    ann_relfrac_amp = 0.2
    ann_relfrac_phase = 0.2
if ann_bounds_method == "plusminus":
    ann_pm_freq = 0.2
    ann_pm_amp = 1
    ann_pm_phase = 0.5
ann_zp_bounds_lower = -10
ann_zp_bounds_upper = 10
# ========== Fitting ==========
# ========== Periodograms ==========
periodograms_lowerbound = "resolution"  # can be set to resolution or explicit value in c/d
periodograms_upperbound = 20

# ========== Plots ==========

figure_subdir = "/figures"
iterative_subdir = "/figures_iterative"
plot_iterative_lcs = True
plot_iterative_pgs = True

# ========== Dual Annealing ==========
frequencies_da_filename = "frequencies_da.csv"
