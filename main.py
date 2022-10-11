# python -B main.py 32 0.75 42
import sys
import numpy as np
from fbm1d import FBM1d
from fbm2d import FBM2d

L = int(sys.argv[1])
H = float(sys.argv[2])
seed = int(sys.argv[3])

np.random.seed(seed)
arr1d = FBM1d(L, H)
arr2d = FBM2d(L, H)

print(arr1d)
print(arr2d)
