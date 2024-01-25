"""
USAGE:
```sh
./plot_search_throughput path/to/hdf5 > results_pc.csv
./plot_search_throughput > results_unif.csv
python plot_search_throughput.py results_pc.csv results_unif.csv
```
"""

import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

# I don't know what a dataframe is and I refuse to learn


def plot_build_times(fname: str):
    n_points = []
    kdt_build_times = []
    forward_times = []
    afftree_build_times = []

    with open(fname) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for row in reader:
            n_points.append(int(row[0]))
            kdt_build_times.append(float(row[1]))
            forward_times.append(float(row[2]))
            afftree_build_times.append(float(row[3]))

    plt.plot(n_points, np.asarray(kdt_build_times) * 1e3, label="kiddo (k-d tree)")
    plt.plot(n_points, np.asarray(forward_times) * 1e3, label="forward tree")
    plt.plot(n_points, np.asarray(afftree_build_times) * 1e3, label="afftree")
    plt.legend()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Construction time (ms)")
    plt.title(f"Scaling of CC structure construction time")
    plt.show()


def plot_query_times(fname: str, title: str):
    n_points = []
    n_tests = []
    afftree_seq_times = []
    afftree_simd_times = []
    kdt_times = []
    forward_seq_times = []
    forward_simd_times = []

    with open(fname) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for row in reader:
            n_points.append(int(row[0]))
            n_tests.append(int(row[1]))
            kdt_times.append(float(row[2]))
            forward_seq_times.append(float(row[3]))
            forward_simd_times.append(float(row[4]))
            afftree_seq_times.append(float(row[5]))
            afftree_simd_times.append(float(row[6]))

    plt.plot(n_points, np.asarray(kdt_times) * 1e9, label="kiddo (k-d tree)")
    plt.plot(
        n_points, np.asarray(forward_seq_times) * 1e9, label="forward tree (sequential)"
    )
    plt.plot(
        n_points, np.asarray(forward_simd_times) * 1e9, label="forward tree (SIMD)"
    )
    plt.plot(
        n_points, np.asarray(afftree_seq_times) * 1e9, label="afftree (sequential)"
    )
    plt.plot(n_points, np.asarray(afftree_simd_times) * 1e9, label="afftree (SIMD)")
    plt.legend()
    plt.semilogy()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Query time (ns)")
    plt.title(title)
    plt.show()


plot_build_times(sys.argv[1])
plot_query_times(sys.argv[2], "Scaling of CC on mixed queries")
plot_query_times(sys.argv[3], "Scaling of CC on all-colliding queries")
plot_query_times(sys.argv[4], "Scaling of CC on non-colliding queries")
