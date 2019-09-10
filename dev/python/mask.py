import numpy as np

A = np.matrix([[0, 0, 0], [0, 5, 0], [0, 0, 0]])

#inp = np.ma.array(A, 
#                   mask=[[1,1,1],[1,0,1], [1,1,1]])
inp = np.ma.masked_equal(A, 0.0)
print(inp)
imask = inp.copy()
imask.mask = ~inp.mask

out = inp
#imask = inp[~inp.mask]
print(imask)
out = out - np.roll(imask, 1, 0) \
            - np.roll(imask, -1, 0) \
            + 4*out \
            - np.roll(imask, 1, 1) \
            - np.roll(imask, -1, 1)
print(out)
