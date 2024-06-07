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

N_FORESTS = 10


def plot_build_times(fname: str):
    n_points = np.genfromtxt(fname, usecols=0, delimiter=",")
    kdt_build_times = np.genfromtxt(fname, usecols=1, delimiter=",")
    forward_times = np.genfromtxt(fname, usecols=2, delimiter=",")
    captree_build_times = np.genfromtxt(fname, usecols=3, delimiter=",")
    forest_build_times = np.genfromtxt(fname, usecols=range(4, 4 + N_FORESTS), delimiter=",")

    plt.plot(n_points, kdt_build_times * 1e3, label="kiddo (k-d tree)")
    plt.plot(n_points, forward_times * 1e3, label="forward tree")

    for (i, btime) in enumerate(forest_build_times.T):
        plt.plot(n_points, btime * 1e3, label=f"forest (T={i + 1})")

    plt.legend()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Construction time (ms)")
    plt.title(f"Scaling of CC structure construction time")
    plt.show()

    plt.plot(n_points, kdt_build_times * 1e3, label="kiddo (k-d tree)")
    plt.plot(n_points, forward_times * 1e3, label="forward tree")
    plt.plot(n_points, captree_build_times * 1e3, label="CAPT")

    plt.legend()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Construction time (ms)")
    plt.title(f"Scaling of CC structure construction time")
    plt.show()



def plot_query_times(fname: str, title: str):
    n_points = []
    n_points = np.genfromtxt(fname, usecols=0, delimiter=",")
    n_tests = np.genfromtxt(fname, usecols=1, delimiter=",")
    kdt_times = np.genfromtxt(fname, usecols=2, delimiter=",")
    forward_seq_times = np.genfromtxt(fname, usecols=3, delimiter=",")
    forward_simd_times = np.genfromtxt(fname, usecols=4, delimiter=",")
    captree_seq_times = np.genfromtxt(fname, usecols=5, delimiter=",")
    captree_simd_times = np.genfromtxt(fname, usecols=6, delimiter=",")
    forest_times = np.genfromtxt(fname, usecols=range(7, 7 + N_FORESTS), delimiter=",")

    plt.plot(n_points, kdt_times * 1e9, label="kiddo (k-d tree)")
    plt.plot(n_points, forward_seq_times * 1e9, label="forward tree (sequential)")
    plt.plot(n_points, forward_simd_times * 1e9, label="forward tree (SIMD)")
    plt.legend(loc="upper left")
    plt.semilogy()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Query time (ns)")
    plt.title(title)
    plt.show()


    plt.plot(n_points, kdt_times * 1e9, label="kiddo (k-d tree)")
    plt.plot(n_points, forward_simd_times * 1e9, label="forward tree (SIMD)")

    plt.plot(n_points, forest_times[:, 0] * 1e9, label=f"forest (SIMD, T={1})")
    plt.plot(n_points, forest_times[:, 4] * 1e9, label=f"forest (SIMD, T={5})")
    plt.plot(n_points, forest_times[:, 9] * 1e9, label=f"forest (SIMD, T={10})")

    plt.legend(loc="upper left")
    plt.semilogy()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Query time (ns)")
    plt.title(title)
    plt.plot(n_points, forward_seq_times * 1e9, label="forward tree (sequential)")
    plt.plot(n_points, forward_simd_times * 1e9, label="forward tree (SIMD)")
    plt.plot(n_points, captree_seq_times * 1e9, label="CAPT (sequential)")
    plt.plot(n_points, captree_simd_times * 1e9, label="CAPT (SIMD)")

    plt.legend(loc="upper left")
    plt.semilogy()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Query time (ns)")
    plt.title(title)
    plt.show()


def plot_mem(fname: str):
    n_points = []
    forward_mem = []
    captree_mem = []

    with open(fname) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for row in reader:
            n_points.append(int(row[0]))
            forward_mem.append(int(row[1]))
            captree_mem.append(int(row[2]))

    plt.plot(n_points, forward_mem, label="Forward tree")
    plt.plot(n_points, captree_mem, label="CAPT")
    plt.semilogy()
    plt.legend()
    plt.xlabel("Number of points in cloud")
    plt.ylabel("Memory used (bytes)")
    plt.title("Memory consumption")
    plt.show()


plot_build_times(sys.argv[1])
plot_query_times(sys.argv[2], "Scaling of CC on mixed queries")
plot_query_times(sys.argv[3], "Scaling of CC on all-colliding queries")
plot_query_times(sys.argv[4], "Scaling of CC on non-colliding queries")
plot_mem(sys.argv[5])
