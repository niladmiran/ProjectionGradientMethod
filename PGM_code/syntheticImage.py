# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:35:00 2021

@author: jacob
"""

import numpy as np
from run_PGM import run_proj_lpball, run_proj_l1ball, IHT
import matplotlib.pyplot as plt
import data_loader

# %% experiment configuration

# Generate synthetic images, and projections
l = 128
sigma = 0.15
A, x_opt, b = data_loader.loadSyntheticImage(l, sigma)

# set the initial guess
x0 = np.zeros(A.shape[1])

# p ball
p = 0.49

# set the radius
radius_lp = np.linalg.norm(x_opt, p) ** p
radius_l1 = np.linalg.norm(x_opt, 1)
radius_l0 = np.count_nonzero(x_opt)

# set the maximum iterations for algorithm and the epsilon optimality
MAX_ITERS = 1e4
tolerance = 1e-7

# %% run algorithm
result_lp = run_proj_lpball(A, x0, b, p, radius_lp, tolerance, MAX_ITERS)
result_l1 = run_proj_l1ball(A, x0, b, radius_l1, tolerance, MAX_ITERS)
result_l0 = IHT(A, x0, b, radius_l0, tolerance, MAX_ITERS)

# %% print the algorithm information
for result in [result_lp, result_l0, result_l1]:
    print("The information of PGM on ", result['name'], " ball")
    print("iter=%3d, time=%.2e, error=%4.2e, obj_value=%4.3e, nonzero=%3d"
          % (result['# iter'],
             result['time'],
             result['residual'][-1],
             result['obj_val'][-1],
             result['nonzero'][-1]))

# %% plot
rec_lp = result_lp['x_opt']
rec_l1 = result_l1['x_opt']
rec_l0 = result_l0['x_opt']

rec_lp = np.where(rec_lp < 1e-10, 0, rec_lp)
rec_l1 = np.where(rec_l1 < 1e-10, 0, rec_l1)
rec_l0 = np.where(rec_l0 < 1e-10, 0, rec_l0)

x_opt = x_opt.reshape(l, l)
rec_lp = rec_lp.reshape(l, l)
rec_l1 = rec_l1.reshape(l, l)
rec_l0 = rec_l0.reshape(l, l)

plt.subplot(221)
plt.imshow(x_opt, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original image')

plt.subplot(222)
plt.imshow(rec_lp, cmap=plt.cm.gray)
plt.title('Lp constrains (p = %.2f)' % p)
plt.axis('off')

plt.subplot(223)
plt.imshow(rec_l1, cmap=plt.cm.gray)
plt.title('L1 constraint')
plt.axis('off')

plt.subplot(224)
plt.imshow(rec_l0, cmap=plt.cm.gray)
plt.title('L0 constraint')
plt.axis('off')

# plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
#                     right=1)

plt.show()
