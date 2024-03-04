# GPgyro
This is an example code to calculate the gyrochronology age for a star using its temperature and rotation period measurements.

## Installation/Setup
1. Download all `.npy` and `GPgyro.np` files
2. Open `GPgyro.np` and change `path_to_files` to the path to all `.npy` files (default is `./`)
3. Make sure the following packages are installed:
   - `jax`, version 0.4.8
   - `numpy`, version 1.21.5
   - `tinygp`, version 0.2.3
   - `arviz`, version 0.15.1
   - `corner`, version 2.2.2
   - `pandas`, version 1.4.4
   - `tqdm`, version 4.64.1
  
## To use the package, do:
```
from GPgyro import *
# create a list of temperatures, periods, and absolute Gaia magnitude (The absolute Gaia magnitude is used to determine whether a star is partically convective or fully convective)
teffs = [5000, 3000, 6000] # list of temperatures 
prots = [20, 100, 15] # list of periods
MG = [5, 10, 7] # list of absolute Gaia magnitude

# create the input list, make sure periods are in log10 scale
X_k = np.array([teffs, np.log10(prots)]).T
MG = np.array(MG)

# calculate gyrochronology age for these stars
age, age_m, age_p = GP_gyro(X_k, MG.T)
```

`age`, `age_m`, `age_p` are the ages, minus uncertainties, and plus uncertainties for these stars
