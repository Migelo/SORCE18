#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import scipy.stats
import common

filename = __file__.split("/")[-1]


# load data
segment = np.load('../66.segment.npy')

# mask
mask = (segment[:, 0] > 5000) & (segment[:, 0] < 6000)
segment = segment[mask]

sorted_segment = segment[np.argsort(segment[:, 1])]
sorted_segment[:, 0] = segment[0, 0]
sorted_segment[1:, 0] += np.cumsum(sorted_segment[:-1, 2])

odf, edges, _ = scipy.stats.binned_statistic(
        segment[:, 0], sorted_segment[:, 1], statistic='mean',
        bins=[5000, 5450, 5850, 5900, 6000])
odf = list(odf)
odf.insert(-1, odf[-1])

# plotting
for i in range(2):
    f, ax = plt.subplots(1)

    ax.plot(sorted_segment[:, 0], np.log10(sorted_segment[:, 1]),
            label='Sorted opacity')
    if i > 0:
        ax.step(edges, np.log10(odf), where='post',
                label='ODF')

    ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')

    ax.set_ylabel(r'log$_{10}$ Opacity [cm$^{-1}$]')
    ax.set_xlim(common.xlim)
    ax.set_xticks([])
    ax.legend(loc='upper left')

    plt.savefig("../../images/{}_{}.{}".format(
            filename[:-3], i, common.pic_format))
