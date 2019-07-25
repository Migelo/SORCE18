#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import common
from scipy import interpolate

filename = __file__.split("/")[-1]

transmission = np.loadtxt('Stroemgren_b.txt')[::-1]
transmission[:, 0] *= 10
transmission[:, 1] /= 100
interp = interpolate.interp1d(transmission[:, 0], transmission[:, 1])
x = np.linspace(4540, 4800, 10)
widths = np.diff(x).tolist() + [x[1] - x[0]]
widths = np.array(widths)
height = np.load('height.npy')
height /= height.max()

widht_trans = widths * interp(x)
widht_trans *= widths.sum() / widht_trans.sum()
x_trans = np.cumsum(widht_trans)
x_trans = np.roll(x_trans, 1)
x_trans[0] = 0
x_trans += x[0]


color = ['C{}'.format(x % 2) for x in range(x.shape[0])]
#common.set_style('3x1')
font = 14
f, ax = plt.subplots(3, sharex=True)
ax[0].set_ylabel('Normalized flux', fontsize=font)
ax[0].bar(x, height, widths, align='edge', color=color)
ax[1].set_ylabel('Transmission curve', fontsize=font)
ax[1].plot(x + widths[0] / 2, interp(x))
ax[1].scatter(x + widths[0] / 2, interp(x))
ax[1].set_ylim(-0.04, ax[1].get_ylim()[1])
ax[1].set_ylim(ax[1].get_ylim()[0], 1.05)
#ax[1].set_xticks(np.arange(0, 1, .2))

ax[2].set_ylabel('Normalized flux', fontsize=font)
ax[2].set_xlabel(r'Wavelength')
ax[2].set_xticks([])
ax[2].bar(x_trans, height, widht_trans, align='edge', color=color)

f.align_ylabels()

plt.savefig("../../images/{}.{}".format(filename[:-3], common.pic_format))
