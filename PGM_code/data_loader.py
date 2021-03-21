#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:37:15 2021

@author: moumatsu
"""

import numpy as np
import utils


def load_data(m, n, s):
    """
    generate synthetic data
    :param m: number of samples
    :param n: number of features
    :param s: number of zero components
    :return:
    """
    A = np.random.randn(m, n)
    x_opt = np.random.rand(n)
    # ind = np.random.choice(n, s)
    # x_opt[ind] = 0
    b = A.dot(x_opt)

    return A, x_opt, b


def loadSyntheticImage(l, sigma):
    proj_operator = utils.build_projection_operator(l, l // 7)
    data = utils.generate_synthetic_data(l)
    proj = proj_operator @ data.ravel()[:, np.newaxis]
    proj += sigma * np.random.randn(*proj.shape)

    proj_operator = proj_operator.toarray()
    data_vec = data.astype(float).ravel()[:, np.newaxis][:, 0]

    return proj_operator, data_vec, proj.ravel()


