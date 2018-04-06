#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import scipy.stats
import common

filename = __file__.split("/")[-1]

# load data
spectra = np.load('../spectra_transmitted.npy')
spectra_odf = np.load('odf_spectra_2b_4_2290.npy')
low = np.loadtxt('../filtr_cut_1')[0, 0]
high = np.loadtxt('filtr_cut_1')[-1, 0]
stretch_factor = np.loadtxt('stretch_factor_2b_5_18284')

# mask to appropriate wavelengths
mask = (spectra_odf[:, 0] > low) & (spectra_odf[:, 0] < high)
spectra_odf = spectra_odf[mask]
mask2 = (spectra[:, 0] > low) & (spectra[:, 0] < high)
spectra = spectra[mask2]

# conver to per wavelength
common.to_wavelength(spectra, True)
common.to_wavelength(spectra_odf, True)

odf, edges, _ = scipy.stats.binned_statistic(
        spectra_odf[:, 0], spectra_odf[:, 1], statistic='mean',
        bins=[5216, 5.46943470e+03, 5.83550280e+03, 5.90589790e+03, 5968])
odf = list(odf)
odf.insert(-1, odf[-1])


# gaussian convolution
#N = 1000
# spectra_con = scipy.ndimage.filters.gaussian_filter1d(spectra[:, 1], N)

# plotting
f, ax = plt.subplots(1)

# left plot
ax.plot(spectra[:, 0], spectra[:, 1], label='High resolution spectrum')
ax.step(edges, odf / stretch_factor, where='post', label='ODF spectrum')
# ax.axvline(x=4200)
# ax.axvline(x=6518)
# ax.axvline(x=6905)
# ax.axvline(x=8740)
# ax.axvline(x=8981)
# ax.axvline(x=9030)

ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
ax.set_xlim(common.xlim)

ax.set_ylabel(r'Normalized transmitted flux')

ax.legend(loc='best')
# right plot
plt.savefig("../../images/{}.{}".format(filename[:-3], common.pic_format))
# os.sys("cp {0}.png ../../images/{0}.png".format(filename[:-3]))
