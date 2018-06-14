import numpy as np
import argparse
from multiprocessing import Pool
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
import common

filename = __file__.split("/")[-1]

default_sub_bin_list = '../sub_bin_list.txt'

parser = argparse.ArgumentParser(description='Calculate errors.')
parser.add_argument('folder', type=str, help='Folder with files.')
parser.add_argument('bins', type=str, help='Bins file.')
parser.add_argument('--nproc', type=int, default=20, help='Number of cpus.')
parser.add_argument('--outfile', type=str, default='results.txt',
                    help='Output filename.')
parser.add_argument('--sb_list', type=str, default=default_sub_bin_list,
                    help='Path to the sub bin file list')
parser.add_argument('--plotting', action='store_true',
                    help='if plotting, results are loaded from a file')
parser.add_argument('--results', type=str,
                    help='if plotting, results are loaded from a file')
args = parser.parse_args()

if args.plotting:
    if args.results is None:
        parser.error('--results is needed with --plotting')


def integrate_bins(spectra):
    spectra_binned = np.zeros(len(bins))
    for i, bin_ in enumerate(bins):
        mask = (spectra[:, 0] > bin_[0]) & (spectra[:, 0] < bin_[1])
        spectra_binned[i] = scipy.integrate.trapz(
            spectra[mask][:, 1] / spectra[mask][:, 0]**2,
            x=spectra[mask][:, 0])
    return spectra_binned


def compare(file_name):
    file_name = '{}/odf_spectra_2b_{}.npy'.format(args.folder, file_name[8:])
    print('{}'.format(file_name.split('/')[-1]))
    raw_data = np.load(file_name)
    binned_data = integrate_bins(raw_data)
    return binned_data / detailed_spectra_binned


# load bins
bins = np.loadtxt(args.bins)
mask = bins[:, 0] >= 1000
mask &= bins[:, 0] < 9000
bins = bins[mask]

# load the detailed spectra
detailed_spectra = np.load('spectra.npy')
detailed_spectra_binned = integrate_bins(detailed_spectra)

# load sub bin combinations
sub_bins = np.loadtxt(args.sb_list, dtype=str)

sub_bin_values = []
for item in sub_bins:
    sub_bin_values.append(np.loadtxt('../sub_bins/{}'.format(item)))


if not args.plotting:
    p = Pool(args.nproc)
    results = p.map(compare, sub_bins)
    results = np.array(results)
    np.save('results', results)
else:
    results = np.load('{}'.format(args.results))

argsorts = []
for i, _ in enumerate(bins):
    argsorts.append(np.argsort(np.abs(results[:, i] - 1)))

# calculate chi2 for all sub bin combinations
chi2 = np.sum((results - 1)**2, axis=1)
chi2_ir = np.sum((results[:, 400:] - 1)**2, axis=1)
chi2_uv = np.sum((results[:, 100:250] - 1)**2, axis=1)

# load the remaining files
kurucz = np.load('odf_spectra/odf_spectra_2b_k.npy')
continuum = np.load('odf_spectra/odf_spectra_cont_only.npy')
kurucz_binned = integrate_bins(kurucz)
continuum_binned = integrate_bins(continuum)


# plotting
colors = ['C{}'.format(x) for x in range(9)]
for plot_number in range(1):
    best_combination = []
    if plot_number == 0:
        f, ax = plt.subplots(3, 1)
    else:
        f, ax = plt.subplots(2, 1)

    # upper panel
    ax[0].set_xlim((1000, 9000))
    ax[0].set_ylim((0, 1))
    for k, bin_ in enumerate(bins):
        sub_bin_to_plot = sub_bin_values[argsorts[k][0]]
        best_combination.append([sub_bins[argsorts[k][0]],
                                results[argsorts[k][0], k],
                                sub_bin_values[argsorts[k][0]]])
        for j, value in enumerate(sub_bin_to_plot):
            if k == 0:
                ax[0].fill_between([bin_[0], bin_[1]], y2=value[0],
                                   y1=value[1], color=colors[j],
                                   label='{}. sub bin'.format(j + 1))
            else:
                ax[0].fill_between([bin_[0], bin_[1]], y2=value[0],
                                   y1=value[1], color=colors[j])
        ax[0].set_ylabel('Sub bin distribution', fontsize=17)
    if plot_number < 1:
        # middle panel
        ax[1].set_ylim((.95, 1.05))
        ax[1].set_xlim((1000, 3600))
        ax[1].set_ylabel('Ratio')
        ax[1].step(bins[:, 0], [1 - abs(x[1] - 1) for x in best_combination],
                   label='best combination')
        ax[1].step(
            bins[:, 0], kurucz_binned / detailed_spectra_binned,
            label='Kurucz')
        ax[1].step(bins[:, 0], detailed_spectra_binned / continuum_binned,
                   label='1/Continuum')
        ax[1].axhline(1)
        # for i, item in enumerate(results[np.argsort(chi2)[:1]]):
        #     ax[1].step(bins[:, 0], item, label=sub_bins[np.argsort(chi2)[i]])
        ax[1].legend(loc='upper right')

        # lower panel
        ax[2].set_ylim((.98, 1.02))
        ax[2].set_xlim((4000, 9000))
        ax[2].set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
        ax[2].set_ylabel('Ratio')
        ax[2].step(bins[:, 0], [1 - abs(x[1] - 1) for x in best_combination],
                   label='best combination')
        ax[2].step(
            bins[:, 0], kurucz_binned / detailed_spectra_binned,
            label='Kurucz')
        ax[2].step(bins[:, 0], detailed_spectra_binned / continuum_binned,
                   label='1/Continuum')
        ax[2].axhline(1)
        # for i, item in enumerate(results[np.argsort(chi2)[:1]]):
        #     ax[1].step(bins[:, 0], item, label=sub_bins[np.argsort(chi2)[i]])
#        ax[2].legend(loc='upper right')

    else:
        # lower panel
        ax[1].set_ylim((.901, 1.1))
        ax[1].set_xlim((1000, 9000))
        ax[1].set_xlabel(u'Wavelength [$\mathrm{\AA}$]')
        ax[1].step(bins[:, 0], [1 - abs(x[1] - 1) for x in best_combination],
                   label='best combination')
        ax[1].step(
            bins[:, 0], kurucz_binned / detailed_spectra_binned,
            label='Kurucz')
        for i, item in enumerate(results[np.argsort(chi2_uv)[:1]]):
            ax[1].step(bins[:, 0], item, label='best UV')
        for i, item in enumerate(results[np.argsort(chi2_ir)[:1]]):
            ax[1].step(bins[:, 0], item, label='best IR')
        ax[1].axhline(1)
        ax[1].legend(loc='upper right')
    f.tight_layout()
#    plt.show()

    plt.savefig(
        "../../images/{}_{}.{}".format(
            filename[:-3], plot_number, common.pic_format))
