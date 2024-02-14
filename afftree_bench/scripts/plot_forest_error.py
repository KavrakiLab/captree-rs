#!/usr/bin/python
"""
USAGE: 
```
python ./plot_error_hist.py path/to/errors.csv
```

Columns must be separated by tabs (`\\t`).
errors.csv must have the number of trees in the forest in column 0, the absolute error in column 1, 
the relative error in column 2, and the exact distance in column 3.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import collections

csv_fname = sys.argv[1]
abs_errs = collections.defaultdict(lambda: [])
rel_errs = collections.defaultdict(lambda: [])
exact_dists = collections.defaultdict(lambda: [])
with open(csv_fname) as f:
    for row in csv.reader(f, delimiter="\t"):
        n_trees = int(row[0])
        abs_err = float(row[1])
        rel_err = float(row[2])
        exact_dist = float(row[3])
        abs_errs[n_trees].append(abs_err)
        rel_errs[n_trees].append(rel_err)
        exact_dists[n_trees].append(exact_dist)

# CDFs

for n_trees, abs_err in abs_errs.items():
    h, edges = np.histogram(abs_err, bins=400)
    cy = np.cumsum(h / np.sum(h))

    plt.plot(edges[:-1], cy, label=f"T={n_trees}")
    # plt.fill_between(edges[:-1], cy, step="pre", alpha=0.4)
plt.xlabel("Absolute distance error (m)")
plt.ylabel("Frequency")
plt.title(f"CDF of forest absolute error distribution")
plt.legend()
plt.show()
