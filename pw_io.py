import config
from datetime import datetime

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
        #f.write(datetime.today.strftime('%Y-%m-%d-%H:%M:%S'))
        for key in cfgdict:
            f.write(f"{key} = {cfgdict[key]} \n")

def format_output(n, e, npts):
    split_n = str(n).split('.')
    split_e = str(e).split('.')
    roundto = 0
    if len(split_n) == 1: # int
        pass
    elif len(split_n) == 2: # float
        error_decimal = split_e[1]
        for i in range(len(error_decimal)):
            if error_decimal[i] != "0":
                roundto = i + npts
                while len(error_decimal) < roundto:
                    error_decimal += "0"
                break

        err_out = error_decimal[roundto-npts:roundto]
        return f"{n:.{roundto}f}({err_out})"


if __name__ == "__main__":
    x = format_output(1.645582, 0.000222, 2)
    pass