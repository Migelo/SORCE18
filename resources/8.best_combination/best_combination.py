import numpy as np
import glob
import matplotlib.pyplot as plt
import natsort
from copy import copy
import common

number_of_sub_bins = 3
sigma = 3
filename = __file__.split("/")[-1]


file_list = natsort.natsorted(
        glob.glob('/scratch/cernetic/testRun/fioss/spectra_long_comparison' \
                  '/{}sub/odf_spectra*Comparison'.format(number_of_sub_bins)))
#for item in file_list:
#    int(item.split('Comparison')[0].split('a')[-1])
sub_bins_path = '/scratch/cernetic/AtmosphericParameters/{0}sub/subBins{0}_'.format(number_of_sub_bins)
cont_only = np.loadtxt(
        '/scratch/cernetic/testRun/fioss/1to5/cont_only_spectraComparison')
kurucz = np.loadtxt(
        '/scratch/cernetic/testRun/fioss/1to5/odf_spectrakComparison')

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def sort_array(array, column):
    array = array[np.argsort(array[:, column])]
    return array

"""Every element of data consists of one Contrast file"""
raw_data = []
for item in file_list:
    raw_data.append(np.loadtxt(item))

"""Perform merging"""
lower_bound, upper_bound, step = 1000, 9000, 10
averaging_bins, data = range(lower_bound, 9000 + step, step), []
for item in raw_data:
    data.append([])
for j, item in enumerate(raw_data):
    averaged_item = []
    for i in range(len(averaging_bins)-1):
        interval = item[(item[:, 0] >= averaging_bins[i]) & (averaging_bins[i+1] > item[:, 0])][:, -1]
        interval = (np.ones(len(interval)) - interval)**2
        averaged_item.append(np.sum(interval))
    data[j] = np.c_[np.array(averaging_bins[:-1]), np.array(averaged_item)]

"""sub_bins contains all chi2 values of 1 sub bins across all Contrast files"""
sub_bins = []
for x in range(len(data[0])):
    sub_bins.append([])
for i in range(len(data)):
    for j in range(len(data[i])):
        sub_bins[j].append(data[i][j])

"""sub_bins contains all values of 1 sub bins across all Contrast files"""
raw_sub_bins = []
for x in range(len(raw_data[0])):
    raw_sub_bins.append([])
for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        raw_sub_bins[j].append(raw_data[i][j])

"""min_indexes contains the index of the minimum value for each sub bin"""
min_indexes = []
for item in sub_bins:
    min_indexes.append([item[np.argmin([x[-1] for x in item])], np.argmin([x[-1] for x in item])])

"""optimal combination"""
optimal_combination = [x[np.argmin(np.abs(1-np.array(x)[:, -1]))][-1] for x in raw_sub_bins]
#optimal_combination =
optimal_combination = np.c_[[x[0][0] for x in raw_sub_bins], optimal_combination]

"""Create best combination"""
best_combination = []
for i, item in enumerate([x[-1] for x in min_indexes]):
    best_combination.append(sub_bins[i][item])
# np.savetxt("chi2_results", best_combination, fmt='%s')

"""Global chi2"""
chi2_list = []
for i, item in enumerate(data):
    chi2_list.append(np.sum((np.ones(len(item)) - np.array(
            [x[-1] for x in item]))**2))
chi2_list = np.array(chi2_list)
chi2_list /= np.min(chi2_list)
chi2_global_results = sort_array(np.c_[file_list, chi2_list], -1)
#np.savetxt('chi2_global_results', chi2_global_results, fmt='%s')


"""Plot"""
sub_bin_names, distribution = [], []
for item in min_indexes:
    sub_bin_names.append(str(item[-1])) # change here
    distribution.append([np.loadtxt(sub_bins_path + sub_bin_names[-1])])

for i, item in enumerate(distribution):
    distribution[i] = [x[0] for x in item[0]] + [1]

new_list = []
for i in distribution:
    new_list.append([])
for i, item in enumerate(distribution):
    for j in range(len(item) - 1):
        new_list[i].append(item[j+1] - item[j])

distribution = new_list

colors = ['C{}'.format(x) for x in range(10)]
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_ylim((0., 1))
ax1.set_xlim((0., len(distribution)))
ax1.set_xticklabels([])
ax1.set_ylabel('Sub bin position and size')
ax1.fill_between(range(len(distribution)), 0, np.array(distribution)[:, 0],
                 label='1. sub bin', facecolor=colors[0])
i, following,  = 0, []
for i in range(len(distribution[0]) - 2):
    if i == 0:
        previous = np.array(distribution)[:, 0]
    else:
        previous = copy(following)
    following = previous + np.array(distribution)[:, i+1]
    ax1.fill_between(range(len(distribution)), previous, following,
                     label=str(i+2) + '. sub bin', facecolor=colors[i+1])
    i += 1
if i == 0:
    ax1.fill_between(range(len(distribution)), np.array(distribution)[:, 0], 1,
                     label='2. sub bin', facecolor=colors[1])
else:
    ax1.fill_between(range(len(distribution)), following, 1,
                     label=str(len(distribution[0])) + '. sub bin',
                     facecolor=colors[i+1])
# sort both labels and handles by labels
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1], loc='lower right')

ax1.set_title("{} sub bins".format(number_of_sub_bins))

ax2.axhline(y=1)
ax2.set_xlim((lower_bound, upper_bound))
ax2.set_ylim((.8,1.05))
ax2.set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
ax2.set_ylabel('Ratio')
#ax2.grid(1)
ax2.plot(optimal_combination[:, 0],
         common.convolve(optimal_combination[:, 1], sigma),
         label='Best combination')
ax2.plot(cont_only[:, 0], common.convolve(1 / cont_only[:, -1], sigma),
         label='Continuum only')
ax2.plot(kurucz[:, 0], common.convolve(kurucz[:, -1], sigma), label='Kurucz')
ax2.legend(loc='lower right')
#ax2.set_xlim(common.xlim)
#plt.savefig('subBins5' + '.pdf')
#plt.savefig(sub_bins_path[-9:-1] + '.pdf')

labels1 = ['{}'.format(x + 1000) for x in np.arange(0, 800, 1, dtype=int)]
ax1.set_xticks(range(len(distribution)), labels1, rotation='vertical')

plt.savefig("../../images/{}.png".format(filename[:-3], common.pic_format))
