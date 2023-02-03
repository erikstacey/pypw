import numpy as np
import matplotlib.pyplot as pl
from models import n_sin_model
import os

n_f = 35

freqs = np.linspace(0.3, 8, n_f)
print(freqs)
amps = np.random.uniform(1, 20, n_f)
phases = np.random.uniform(0, 1, n_f)
zp = 0.3

xnorm = np.linspace(0,30, 30000)
ynorm = n_sin_model(xnorm, *freqs, *amps, *phases, zp)
errs = np.ones(len(xnorm))

savedir = f"{os.getcwd()}/test_dirs/{n_f}_synth_equalspace/"
os.makedirs(savedir, exist_ok=True)
os.chdir(savedir)

freqs_file = "inp_freqs.csv"
with open(freqs_file, 'w') as f:
    for i in range(len(freqs)):
        f.write(f"{freqs[i]},{amps[i]},{phases[i]}\n")

data_file = f"{n_f}_f_synthdata"

with open(data_file+"_equalspaced.dat", 'w') as f:
    for i in range(len(xnorm)):
        f.write(f"{xnorm[i]} {ynorm[i]} {errs[i]}\n")

