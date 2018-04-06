#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import common
import matplotlib as mpl

filename = __file__.split("/")[-1]

# load data
non_odf = np.loadtxt('10A1subComparison')
kurucz_odf = np.loadtxt('odf_spectrakComparison')
sub_bins = np.loadtxt('66.ins_k')

# mask
non_odf = common.mask(1000, 9000, non_odf)
kurucz_odf = common.mask(1000, 9000, kurucz_odf)
sub_bins = common.mask(1000, 9000, sub_bins)
# calculate mean opacities
odf, edges, _ = scipy.stats.binned_statistic(
        sub_bins[:, 0], sub_bins[:, 2], statistic='mean',
        bins=np.arange(1000, 9000, 10))


# mask to same wavelengths
kurucz_odf_mask = kurucz_odf[:, 0] > non_odf[0, 0]
kurucz_odf_mask = kurucz_odf_mask & (kurucz_odf[:, 0] < non_odf[-1, 0])
kurucz_odf = kurucz_odf[kurucz_odf_mask]

# plot
f, ax = plt.subplots(2, 1)

ax[0].set_title('Ratio of ODF spectra to detailed spectra',
                horizontalalignment='center', fontsize=18)
ax[0].set_ylabel('Ratio')
ax[0].plot(kurucz_odf[:, 0], kurucz_odf[:, 2], label='Kurucz sub bins')
ax[0].plot(non_odf[:, 0], non_odf[:, 2], label='Mean')
ax[0].legend(loc='lower right')

ax[1].step(sub_bins[:, 0], np.log10(sub_bins[:, 2]), where='post',
           label='Kurucz sub bins')
ax[1].step(edges[:-1], np.log10(odf), where='post',
           label='Mean')
ax[1].set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
ax[1].set_xlim((3005, 3075))
ax[1].set_ylabel(r'log$_{10}$ Opacity [cm$^{-1}$]')
ax[1].set_ylim((-8.2, -2.5))

# ax[1].legend(loc='lower right')

# second plot in the lower plot
#low, high = 4005, 4055
#ax_1_0 = plt.axes([.7, .125, .22, .22])
#ax_1_0.step(sub_bins[:, 0], np.log10(sub_bins[:, 2]), where='post',
#           label='Kurucz sub bins')
#ax_1_0.step(edges[:-1], np.log10(odf), where='post', label='mean')
#
#ax_1_0.set_xticks([])
#ax_1_0.set_yticks([])

plt.savefig("../../images/{}.{}".format(filename[:-3], common.pic_format))
