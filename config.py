class config_state():

# ========== Main/Most relevant ==========
    version = "2.0.2"
    #  working_dir sets the directory which contains the light curve data file
    #  target_file sets the light curve data file
    #working_dir = "/Users/erikstacey/main/projects/plaskett_photometry/hd47129_analysis_for_thesis/corot"
    #target_file = "corot_data_extracted.txt"
    working_dir = "/Users/erikstacey/main/projects/plaskett_photometry/hd47129_analysis_for_thesis/pdc"
    target_file = "HD47129_thresh9_lc.txt"
    #working_dir = "/Users/erikstacey/main/projects/plaskett_photometry/hd47129_analysis_for_thesis/sap"
    #target_file = "HD47129_thresh9_lc_sap.txt"

    # passthrough to np.loadtxt
    cols = [0, 1, 2]
    delimiter = ' '
    skiprows = 1

    # dtype can be "flux", "mag", "mmag"
    dtype = "flux"

    # file_time_offset (float) is the jd0 of the time measurements in the light curve. For example, TESS data is usually
    # HJD - 2457000, so this would be set to 2457000
    # time_offset (float) is subtracted from the timestamps during pre-processing. This is usually used to set a specific
    # JD0 for measurements of phases of periodic components
    time_offset = 1468.27
    #time_offset = 54748.44921481335
    file_time_offset = 2457000
    #file_time_offset = 2400000
    jd0 = file_time_offset+time_offset  # not set by user - appropriately set file_time_offset and time_offset
    time_unit = "BJD"  # used for labelling. TESS is HJD, CoRoT is BJD

    n_f = 50  # Int / The maximum number of frequencies permitted. %todo: combine n_f and cutoff_iteration settings

    peak_selection = "slf" # string / can be highest, bin, slf
    bin_highest_override = 10 # Int / the number of frequencies selected by highest amplitude with no additional criteria
    averaging_bin_radius = 0.5  # Float / The size of the box used in periodogram box averages
    cutoff_iteration = 50  # Int / The maximum number of frequencies to test for significance during peak selection.
    # This isn't used in the default SLF peak selection, which checks the entire array.
    cutoff_sig = 3.0  # the peak selection significance criterion
    multi_fit_zp = True  # Bool / include the zero point as a free parameter in complete variability model fits
    prevent_frequencies_within_resolution = True  # Bool / Prevent the selection of close frequencies within 1.5/delta T

    multi_fit_type = "lm"  # "anneal", "lm", "scipy". Sets fitting engine to use. Anneal and SciPy are currently out of date.
    residual_model_generation = "mf" # can be sf, mf. Controls which model is used to generate residual periodogram
    boundary_warnings = 0.05  # Float / If an optimized parameter is within this fraction of a boundary, prints a warning
    clean_existing = True  # Bool / Cleans out output directory at the start

    sig_method = "poly"  # Deprecated - All three sig methods are used and output
    poly_order = 3  # Deprecated

# ========== Preprocessing ==========
    # sets the units to convert the data to prior to analysis.
    target_dtype = "flux"  # allowed values: mag, mmag, flux (only if data originally in flux)
    strip_gaps = True  # bool / Strip points around gaps in the time series
    gap_size = 0.5  # Float / Minimum size of gaps, in days, to strip points from around
    strip_gaps_points = 50  # Int, number of points to strip on either side of each gap, and at the start and end of data

# ========== Console ==========
    quiet = False  # Bool / If false, silences a lot of terminal messages

# ========== Output Handler Settings ==========
    output_in_mmag = True  # Bool / Converts plots and data to differential mmag before output.
    # Only set True if data is in flux.

    main_output = "results"  # string / Main output directory
    lcs_output = "lightcurves"  # string / Lightcurves are stored in this subdirectory in the main output
    pgs_output = "periodograms"  # string / Periodograms are stored in this subdirectory in the main output
    misc_output = "supplemental"  # string / Supplemental output data  are stored in this subdirectory in the main output

    lc_xlabel = f"Time [{time_unit} - {jd0}]"  # String / x label of light curves
    pg_xlabel = "Frequency [c/d]"  # String / x label of periodograms

    #  set appropriate y labels for the output data type
    if output_in_mmag:
        lc_ylabel = f"Flux [mmag]"
        pg_ylabel = "Amplitude [mmag]"
    else:
        lc_ylabel = f"Flux [{target_dtype}]"
        pg_ylabel = f"Amplitude [{target_dtype}]"

    preprocessed_lightcurve_fname = "pp_lc.csv"  # string / name of light curve to save following pre-processing
    frequencies_fname = "frequencies.csv"  # string / name of output file for frequencies
    save_supp_data = True  # bool / Save auxiliary data to misc_output subdirectory



    # ========== Frequency Selection ==========
    freq_selection_method = "highest" # can be highest, or averaged

    # ========== Fitting ==========
    phase_fit_rejection_criterion = 0.1
    # ========== LM ==========
    # freq bounds use coefficients
    freq_bounds_lower_coef = 0.8
    freq_bounds_upper_coef = 1.2
    # amp bounds use coefficients
    amp_bounds_lower_coef = 0.8
    amp_bounds_upper_coef = 1.2
    # phase bounds set explicitly
    phase_bounds_lower = -100
    phase_bounds_upper = 100

    # ========== Periodograms ==========
    periodograms_lowerbound = "resolution"  # can be set to resolution or explicit value in c/d
    periodograms_upperbound = 10


    # ========== Periodograms ====
    pg_x_axis_label = "Frequency [c/d]"
    pg_y_axis_label = f"Amplitude [{target_dtype}]"
    pg_xlim = [0, 5]

    # ========== Dual Annealing ==========
    frequencies_da_filename = "frequencies_da.csv"

config = config_state()