#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:01:41 2018

@author: cernetic
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import common

parser = argparse.ArgumentParser(description='generate plot of best sub bins')
parser.add_argument('results', type=str, help='input file')
args = parser.parse_args()

filename = __file__.split("/")[-1]

data = {}
sub_bins = sorted(
    np.loadtxt('../sub_bin_list.txt', dtype=str))
results = np.genfromtxt(
    args.results, dtype=[('name', 'U10'), ('value', 'f8')],
    usecols=np.arange(2))
for item in sub_bins:
    if '{}'.format(item[8:]) not in results['name']:
        print(item)
        continue
results['value'] = 1 - np.abs(1 - results['value'])
argsort = np.argsort(1 - results['value'])
results = results[argsort]

# find best for each number of sub-bins
temp = []
for i in range(1, 7):
    j = 0
    for item in results:
        if item['name'][0] == '{}'.format(i):
            # temp[item['name']] = item['value']
            temp.append(item)
            print(item)
            j += 1
            if j > 10:
                break


sub_bin_values = []
for item in results['name']:
    item = '../sub_bins/sub_bin_{}'.format(item)  # sub_bin_4_72
    sub_bin_values.append(np.loadtxt(item))

for item in sub_bin_values[:5]:
    print(set(item.flatten()))


colors = ['C{}'.format(x) for x in range(9)]
for ijk in range(2):
    f, ax = plt.subplots()
    for i, item in enumerate(sub_bin_values):
        for j, value in enumerate(item):
            if i == 0:
                ax.fill_between([i, i + 1], y2=value[0], y1=value[1],
                                color=colors[j],
                                label='{}. sub bin'.format(j + 1))
            else:
                ax.fill_between([i, i + 1], y2=value[0], y1=value[1],
                                color=colors[j])
    ax.legend(loc='lower right')
    ax.set_ylim((0, 1))
    if ijk > 0:
        ax.set_xlim(0, 50)

    ax2 = ax.twinx()
    ax2.step(np.arange(len(sub_bin_values)),
             results['value'], c='grey', label='Accuracy')
    ax.set_ylabel('Sub bin distribution')
    ax2.set_ylim(.2, ax2.get_ylim()[-1])
    if ijk > 0:
        ax2.set_ylim((.985, 1.))
    else:
        ax2.set_ylim((ax2.get_ylim()[0], ax.get_ylim()[1]))
    ax2.legend(loc='upper right')
    ax.set_xticks([])
    ax2.set_xticks([])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')

    plt.savefig("../../images/{}_{}.{}".format(
                filename[:-3], ijk, common.pic_format))


# temp = np.array(temp, dtype=[('name', 'U10'), ('value', 'f8')])

# f, ax = plt.subplots(1, figsize=(10, 10 / 1.61))
# for i, item in enumerate(temp[::-1]):
#     for j, value in enumerate(item[:-1]):
#         ax.fill_between([i, i + 1], value, item[j + 1], color=colors[j])
# ax2 = ax.twinx()
# ax2.plot(range(len(sub_bin_values)), results['value'][::-1], c='k')
# plt.show()
