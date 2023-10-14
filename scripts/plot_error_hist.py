import csv
import matplotlib.pyplot as plt
import sys

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

# PDFs

plt.hist(rel_errs, bins=400, density=True, stacked=True)
plt.xlabel("Relative distance error")
plt.ylabel("Frequency")
plt.title(f"PDF of relative error distribution for {csv_fname}")
plt.show()


plt.hist(abs_errs, bins=400, density=True, stacked=True)
plt.xlabel("Absolute distance error")
plt.ylabel("Frequency")
plt.title(f"PDF of absolute error distribution for {csv_fname}")
plt.show()


# CDFs

plt.hist(
    rel_errs,
    bins=400,
    density=True,
    cumulative=True,
    label="CDF",
    histtype="step",
    alpha=0.8,
    color="k",
)

plt.xlabel("Relative distance error")
plt.ylabel("Frequency")
plt.title(f"CDF of relative error distribution for {csv_fname}")
plt.show()


plt.hist(
    abs_errs,
    bins=400,
    density=True,
    cumulative=True,
    label="CDF",
    histtype="step",
    alpha=0.8,
    color="k",
)

plt.xlabel("Absolute distance error")
plt.ylabel("Frequency")
plt.title(f"CDF of Absolute error distribution for {csv_fname}")
plt.show()
