import splinefit as sf
import numpy as np

x = np.array([0, 1, 2, 3, 4])
x_ans = np.array([0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4])
x = sf.fitting.refine(x)
assert(np.all(np.isclose(x, x_ans)))
