#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import common

filename = __file__.split("/")[-1]

data_1 = np.loadtxt('./odf_spectra3qComparison')
data_2 = np.loadtxt('./odf_spectra3eComparison')
data_3 = np.loadtxt('./odf_spectra2aComparison')
data_4 = np.loadtxt('./cont_only')

f, ax = plt.subplots(1)

sigma = 10

ax.plot(data_1[:, 0], common.convolve(data_1[:, 2], sigma),
        label='50%, 37.5%, 12.5%')
ax.plot(data_2[:, 0], common.convolve(data_2[:, 2], sigma),
        label='12.5%, 75%, 12.5%')
ax.plot(data_3[:, 0], common.convolve(data_3[:, 2], sigma),
        label='87.5%, 12.5%')
ax.plot(data_4[:, 0], common.convolve(data_4[:, 2], sigma),
        label='1 / Continuum only')
ax.axhline(y=1)

ax.set_xlim((3750, 8900))

ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
ax.set_ylabel('Ratio of ODF spectra to continuum only')
ax.set_xlim((1100, 8900))
ax.set_xlim((3750, 8900))
ax.set_ylim((.4, 1.05))

legend = ax.legend(loc='best')

plt.savefig("../../images/{}.{}".format(filename[:-3], common.pic_format))
