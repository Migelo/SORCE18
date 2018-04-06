import numpy as np
from scipy import stats

filename = '10.segment.npy'

data = np.load(filename)
data = data[(data[:, 0] > 4010) & (data[:, 0] < 4020)]
binned_data, bin_edges, _ = stats.binned_statistic(data[:, 0], data[np.argsort(data[:, 1])][:, 1])
binned_data = list(binned_data)
binned_data.insert(-1, binned_data[-1])

np.save('{}.sorted'.format(filename[:-4]), np.column_stack((data[:, 0], data[np.argsort(data[:, 1])][:, 1])))
np.savetxt('sub_bins', np.column_stack((bin_edges, binned_data)))

