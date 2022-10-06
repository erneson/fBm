# python -B main 32 0.75 42
import sys
import numpy as np
from fbm2d import FBM2d

L = int(sys.argv[1])
H = float(sys.argv[2])
seed = int(sys.argv[3])

np.random.seed(seed)
arr = FBM2d(L, H)

print(arr)
