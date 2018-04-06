import numpy as np
from scipy import interpolate
import argparse

parser = argparse.ArgumentParser(description='Apply filter function to\
                                 spectra.')
parser.add_argument('spectra', type=str, help='Spectra.')
parser.add_argument('filter', type=str, help='Filter function.')
parser.add_argument('suffix', type=str, help='Output filename suffix.')
args = parser.parse_args()

# load everything and mask
if args.spectra[-4:] == '.npy':
    spectra = np.load(args.spectra)
    args.spectra = args.spectra[:-4]
else:
    spectra = np.loadtxt(args.spectra)
filtr = np.loadtxt(args.filter)
wavelength_mask = (spectra[:, 0] >= filtr[0, 0]) & (spectra[:, 0] <= filtr[-1, 0])
spectra = spectra[wavelength_mask]

# create an interpolation object and interpolate the transmission
# function to the fine wavelength grid
interp_object = interpolate.interp1d(filtr[:, 0], filtr[:, 1])
interpolated_filtr = interp_object(spectra[:, 0])

# apply the interpolated transmission function to the calculated spectra
spectra[:, 1] *= interpolated_filtr
# transmitted_spectra = np.column_stack([spectra[:, 0], transmitted_intensity])

# save data
np.save('spectra_{}'.format(args.suffix), spectra)
