#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import common

filename = __file__.split("/")[-1]

# load data
spectra = np.load('spectra_transmitted.npy')
opacity = np.load('../66.segment.npy')


# convert to per wavelength
common.to_wavelength(spectra, True)

# gaussian convolution
N = 5000
spectra_con = scipy.ndimage.filters.gaussian_filter1d(spectra[:, 1], N)

# plotting
for i in range(2):
    if i == 0:
        f, ax = plt.subplots(1, 1)
    if i == 1:
        f, ax = plt.subplots(2, 1)

    # first plot
    if i == 0:
        ax.plot(spectra[:, 0], spectra[:, 1], label='Full')
#        ax.plot(spectra[:, 0], spectra_con, label='Convolved')

        ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
        ax.set_xlim(common.xlim)

        ax.set_ylabel(r'Normalized flux')

    # second plot
    if i == 1:
        # upper plot
        ax[0].plot(opacity[:, 0], np.log10(opacity[:, 1]))
        ax[0].axvline(x=5450)
        ax[0].axvline(x=5460)

        ax[0].set_ylabel(r'log$_{10}$ Opacity [cm$^{-1}$]')

        # lower plot
        ax[1].plot(opacity[:, 0], np.log10(opacity[:, 1]))

        ax[1].set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
        ax[1].set_xlim((5450, 5460))

        ax[1].set_ylabel(r'log$_{10}$ Opacity [cm$^{-1}$]')
        ax[1].set_ylim((-10.25, -4))

    plt.savefig("../../images/{}_{}.{}".format(
            filename[:-3], i, common.pic_format))

