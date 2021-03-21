#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:20:52 2021

@author: moumatsu
"""

import numpy as np


def obj(lamb, weight, y, radius):
    """
    The objective function for solving the weighted l1 ball projection problem
    which is exactly the Lagrangian of the primal problem

    Parameters
    ----------
    lamb : float
        the initial guess, it is the solution of the last iterate problem.
    weight : array, (n, )
        weight vector, nonnegative.
    y : array, (n, )
        the vector to be projected, nonnegative.
    radius : float
        the radius of the weight l1 ball, greater than 0.

    Returns
    -------
    TYPE float
        value of the Lagriange function.

    """
    return np.sum(weight.dot(np.maximum(y - lamb * weight, 0))) - radius


def bisection(weight, y, radius, lamb):
    """
    Use bisection to solve the weighted l1 ball projection problem
    the objective function is given in `obj` function,
    the final solution is given by
    x[i] = max(y[i] - lamb*weight[i], 0)

    Parameters
    ----------
    weight : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    lamb : TYPE
        DESCRIPTION.

    Returns
    -------
    x_opt : TYPE
        DESCRIPTION.
    lamb : TYPE
        DESCRIPTION.

    """
    tolerance = 1e-10  # precision for bisection method
    low = 0

    act_ind = np.where(abs(weight) > 0)[0]
    high = max(y[act_ind] / weight[act_ind])
    value_of_high = obj(high, weight, y, radius)
    value_of_low = obj(low, weight, y, radius)

    assert value_of_high * value_of_low < 0, "The sign must not be the same"

    while True:
        value_of_lamb = obj(lamb, weight, y, radius)

        if abs(value_of_lamb) <= tolerance:
            break

        # bisection method
        if value_of_lamb < 0:
            high = lamb
        else:
            low = lamb

        if abs(high - low) <= tolerance:
            # print("A BAD solution!, the value is %.2e" % abs(value_of_lamb))
            break

        lamb = (high + low) / 2

    x_opt = np.maximum(y - lamb * weight, 0)

    return x_opt, lamb


def sortBased(weight, y, radius):
    """
    Weighted generalization of the sort based algorithm.
    Algorithm 2 of 'Efficient Projection Algorithms onto the Weighted l1 Ball'

    :param weight:
        weight
    :param y:
        the vector to be projected
    :param radius:
        radius of the weighted l1 ball
    :return:
        the projection of y onto the weighted ball defined by weight and radius
    """
    # get rid of tiny weighte
    act_ind = np.where(abs(weight) > 1e-15)[0]
    d = len(act_ind)

    # step two
    pair_seq = zip(y[act_ind], weight[act_ind])

    # step 3, sort
    pair_seq = sorted(pair_seq, key=lambda x: x[0] / x[1])

    z_seq = np.array([x[0]/x[1] for x in pair_seq])
    w_sorted_seq = np.array([x[1] for x in pair_seq])

    for i in range(d-1):
        if obj(z_seq[i], weight, y, radius) * obj(z_seq[i+1], weight, y, radius) < 0:
            break

    numerator = np.dot(w_sorted_seq[i+1:]**2, z_seq[i+1:]) - radius
    dedominator = np.sum(w_sorted_seq[i+1:])

    lamb = numerator / dedominator
    value_of_lamb = obj(lamb, weight, y, radius)
    print("the value is %.2e" % abs(value_of_lamb))

    x_opt = np.maximum(y - weight * lamb, 0)

    return x_opt, lamb













