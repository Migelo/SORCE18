#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import common

#common.set_style('1x1')
filename = __file__.split("/")[-1]

spectra_transmitted = np.load('./spectra_transmitted.npy')
spectra_cont_transmitted = np.load('./spectra_cont_transmitted.npy')

f, ax = plt.subplots(1)

ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
ax.set_ylabel('Normalized flux')
ax.set_xlim(4475, 4900)

ax.plot(spectra_transmitted[:, 0],
        spectra_transmitted[:, 1] / spectra_cont_transmitted[:, 1].max(),
        label=r'Full linelist', linewidth=.3)
ax.plot(spectra_cont_transmitted[:, 0],
        spectra_cont_transmitted[:, 1] / spectra_cont_transmitted[:, 1].max(),
        label=r'Continuum only')
ax.legend(loc='upper right')

plt.savefig("../../images/{}.{}".format(filename[:-3], common.pic_format))
