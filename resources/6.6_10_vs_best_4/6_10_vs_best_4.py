#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import common

filename = __file__.split("/")[-1]

data_1 = np.loadtxt('./odf_spectra10Comparison')
data_2 = np.loadtxt('./odf_spectra9Comparison')
data_3 = np.loadtxt('./odf_spectra8Comparison')
data_4 = np.loadtxt('./odf_spectra7Comparison')
data_5 = np.loadtxt('./odf_spectra6Comparison')
data_6 = np.loadtxt('./odf_spectra3c5025Comparison')

for i in range(2):
    f, ax = plt.subplots(1)

    sigma = 10

    ax.plot(data_1[:, 0], common.convolve(data_1[:, 2], sigma),
            label='10 uniform sub bins')
    ax.plot(data_2[:, 0], common.convolve(data_2[:, 2], sigma),
            label='9 uniform sub bins')
    ax.plot(data_3[:, 0], common.convolve(data_3[:, 2], sigma),
            label='8 uniform sub bins')
    ax.plot(data_4[:, 0], common.convolve(data_4[:, 2], sigma),
            label='7 uniform sub bins')
    ax.plot(data_5[:, 0], common.convolve(data_5[:, 2], sigma),
            label='6 uniform sub bins')
    if i > 0:
        ax.plot(data_6[:, 0], common.convolve(data_6[:, 2], sigma),
            label='4 nonuniform sub bins', lw=5)
    ax.axhline(y=1)

    ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
    ax.set_ylabel('Ratio of ODF- to high resolution-spectrum')
    ax.set_xlim((3800, 8900))
    ax.set_ylim((.95, 1.005))

    ax.legend(loc='best')

    plt.savefig("../../images/{}_{}.{}".format(filename[:-3], i, common.pic_format))
