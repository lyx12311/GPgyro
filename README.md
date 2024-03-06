# gpgyro
This is an example code to calculate the gyrochronology age for a star using its temperature and rotation period measurements.

## Installation/Setup
```
git clone https://github.com/lyx12311/GPgyro.git
cd GPgyro
python -m pip install -e .
```
  
## To use the package, do:
```python
from gpgyro import *
import numpy as np

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
