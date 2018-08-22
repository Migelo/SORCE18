#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.ndimage.filters

plt.style.use(['my_dark_background', 'presentation_169'])

# physical constants
c = 2.99792458e+10

# plotting settings
xlim = (4950, 6050)
pic_format = 'pdf'


def to_wavelength(nessy, normalize=False):
    nessy[:, 1] = 1e-8 * nessy[:, 1] * c / (nessy[:, 0] * 1e-8)**2
    if normalize:
        nessy[:, 1] /= nessy[:, 1].max()


def convolve(array, sigma):
    return scipy.ndimage.filters.gaussian_filter1d(array, sigma)


def mask(low, high, array):
    mask = (array[:, 0] > low) & (array[:, 0] < high)
    array = array[mask]
    return array
