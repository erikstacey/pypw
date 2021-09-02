import config

def save_csv_flist(filename, freqs, amps, phases, ferr, aerr, perr, sig):
    if not config.quiet:
        print(f"Writing {len(freqs)} freqs to {filename}")
    with open(config.working_dir+"/"+filename, 'w') as f:
        f.write(f"{len(freqs)}\n")
        for i in range(len(freqs)):
            f.write(f"{freqs[i]},{amps[i]},{phases[i]},{ferr[i]},{aerr[i]},{perr[i]},{sig[i]}\n")

def save_config(filename):
    cfgdict = config.__dict__
    with open(config.working_dir+f"/{filename}", 'w') as f:
        for key in cfgdict:
            f.write(f"{key} = {cfgdict[key]} \n")

