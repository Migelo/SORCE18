#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import common

filename = __file__.split("/")[-1]

# load data
opacity = np.loadtxt('dp67_9000A.asc')
segment = np.load('../66.segment.npy')

sorted_segment = segment[np.argsort(segment[:, 1])][:, 1]
odf, edges, _ = scipy.stats.binned_statistic(
        segment[:, 0], sorted_segment, statistic='mean')
odf = list(odf)
odf.insert(-1, odf[-1])
odf_2, edges_2, _ = scipy.stats.binned_statistic(
        segment[:, 0], sorted_segment, statistic='mean', bins=1)
odf_2 = list(odf_2)
odf_2.insert(-1, odf_2[-1])

# plotting
for i in range(5):
    f, ax = plt.subplots(1)

    if i < 2:
        ax.plot(segment[:, 0], np.log10(segment[:, 1]),
                label='High resolution opacity')
    if i > 0:
        ax.plot(segment[:, 0], np.log10(sorted_segment),
                label='Sorted opacity', c='C1')
    if i > 2:
        ax.step(edges, np.log10(odf), where='post', label='ODF')
#        ax.set_title('Example with 10 uniform sub bins')
    if i > 3:
        ax.step(edges_2, np.log10(odf_2), where='post', label='Mean', c='C2')

    ax.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')

    if i > 1:
        ax.set_xticks([])
        ax.set_xlabel('Wavelength index')

    ax.set_ylabel(r'log$_{10}$ Opacity [cm$^{-1}$]')
#    ax.set_ylim((4010, 4020))
#    ax.set_ylim((-17.3, -14.5))
#    ax.set_xticks(range(9490, 9515, 5))
    ax.legend(loc='best')

    plt.savefig("../../images/{}_{}.{}".format(filename[:-3], i,
                common.pic_format))
