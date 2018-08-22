import numpy as np

bin_edges = []
best_combination = []

a = np.array([x[2][1:, 0] for x in best_combination if len(x[2]) == 4])
unique, counts = np.unique(a, return_counts=True)
dict(zip(unique, counts))
