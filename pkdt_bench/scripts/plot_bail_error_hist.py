#!/usr/bin/python

"""
USAGE: 
```
python ./plot_bail_error_hist.py path/to/errors.csv
```

Columns must be separated by tabs (`\\t`).
errors.csv must have the bail height in column 0, anything in column 1, true distance in column 2, 
estimated distance in column 3, and relative error in column 4.
"""

import csv
import matplotlib.pyplot as plt
import sys

csv_fname = sys.argv[1]
rel_errs = {}
true_dists = {}
abs_errs = {}
approx_dists = {}
with open(csv_fname) as f:
    for row in csv.reader(f, delimiter="\t"):
        bail_height = int(row[0])

        if bail_height not in rel_errs.keys():
            rel_errs[bail_height] = []
            true_dists[bail_height] = []
            abs_errs[bail_height] = []
            approx_dists[bail_height] = []

        rel_errs[bail_height].append(float(row[4]))
        true_dists[bail_height].append(float(row[2]))
        abs_errs[bail_height].append(float(row[3]) - float(row[2]))
        approx_dists[bail_height].append(float(row[3]))

# PDFs

# for bail_height in rel_errs.keys():
#     plt.hist(rel_errs[bail_height], bins=400, density=True, stacked=True)
#     plt.xlabel("Relative distance error")
#     plt.ylabel("Frequency")
#     plt.title(
#         f"PDF of relative error distribution for {csv_fname} at bail height {bail_height}"
#     )
#     plt.show()

#     plt.hist(abs_errs[bail_height], bins=400, density=True, stacked=True)
#     plt.xlabel("Absolute distance error")
#     plt.ylabel("Frequency")
#     plt.title(
#         f"PDF of absolute error distribution for {csv_fname} at bail height {bail_height}"
#     )
#     plt.show()


# CDFs

for bail_height in rel_errs.keys():
    plt.hist(
        abs_errs[bail_height],
        bins=400,
        density=True,
        cumulative=True,
        label=bail_height,
        histtype="step",
        alpha=0.8,
    )

plt.xlabel("Absolute distance error")
plt.ylabel("Frequency")
plt.title(
    f"CDF of Absolute error distribution for {csv_fname} at assorted bail heights"
)
plt.legend()
plt.show()
