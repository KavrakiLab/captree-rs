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
    plt.hist(
        abs_err,
        bins=400,
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.8,
        label=f"T={n_trees}",
    )

plt.xlabel("Absolute distance error")
plt.ylabel("Frequency")
plt.title(f"CDF of absolute error distribution for {csv_fname}")
plt.legend()
plt.show()


for n_trees, rel_err in rel_errs.items():
    plt.hist(
        rel_err,
        bins=400,
        density=True,
        cumulative=True,
        histtype="step",
        alpha=0.8,
        label=f"T={n_trees}",
    )

plt.xlabel("Relative distance error")
plt.ylabel("Frequency")
plt.title(f"CDF of relative error distribution for {csv_fname}")
plt.legend()
plt.show()

# Scatters

for n_trees, abs_err in abs_errs.items():
    plt.scatter(
        exact_dists[n_trees],
        np.sum(np.asarray([exact_dists[n_trees], abs_err]), axis=0),
        alpha=0.8,
        label=f"T={n_trees}",
    )
plt.xlabel("Exact distance")
plt.ylabel("Estimated distance")
plt.title(f"Distance estimation results")
plt.legend()
plt.show()
