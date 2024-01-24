#!/usr/bin/python

"""
USAGE: 
```
python ./plot_error_hist.py path/to/errors.csv
```

Columns must be separated by tabs (`\\t`).
errors.csv must have anything in column 0, true distance in column 1, estimated distance in column 
2, and relative error in column 3.
"""

import csv
import matplotlib.pyplot as plt
import sys
import numpy as np

csv_fname = sys.argv[1]
rel_errs = []
true_dists = []
abs_errs = []
approx_dists = []
with open(csv_fname) as f:
    for row in csv.reader(f, delimiter="\t"):
        rel_errs.append(float(row[3]))
        true_dists.append(float(row[1]))
        abs_errs.append(float(row[2]) - float(row[1]))
        approx_dists.append(float(row[2]))

h, edges = np.histogram(abs_errs, bins=400)
cy = np.cumsum(h / np.sum(h))

plt.plot(edges[:-1], cy)
plt.fill_between( edges[:-1], cy, step="pre", alpha=0.4)
plt.xlabel("Absolute distance error (m)")
plt.ylabel("Frequency")
plt.title(f"CDF of forward tree absolute error distribution")
plt.show()
