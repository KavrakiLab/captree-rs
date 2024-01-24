"""
USAGE: 
```
python ./plot_affordance_time.py path/to/construction_time.csv path/to/query_time.csv
```
"""

import sys
import csv
import matplotlib.pyplot as plt

construct_fname = sys.argv[1]
query_fname = sys.argv[2]

n_points = []
construct_kdt_times = []
construct_forward_times = []
construct_afftree_times = []
with open(construct_fname) as f:
    for row in csv.reader(f, delimiter=","):
        n_points.append(int(row[0]))
        construct_kdt_times.append(float(row[1]) * 1e3)
        construct_forward_times.append(float(row[2]) * 1e3)
        construct_afftree_times.append(float(row[3]) * 1e3)

query_kdt_times = []
query_forward_seq_times = []
query_forward_simd_times = []
query_afftree_seq_times = []
query_afftree_simd_times = []

with open(query_fname) as f:
    for row in csv.reader(f, delimiter=","):
        query_kdt_times.append(float(row[1]) * 1e9)
        query_forward_seq_times.append(float(row[2]) * 1e9)
        query_forward_simd_times.append(float(row[3]) * 1e9)
        query_afftree_seq_times.append(float(row[4]) * 1e9)
        query_afftree_simd_times.append(float(row[5]) * 1e9)

plt.plot(n_points, query_kdt_times, label="k-d tree (kiddo)")
plt.plot(n_points, query_forward_seq_times, label="forward tree, sequential")
plt.plot(n_points, query_forward_simd_times, label="forward tree, SIMD")
plt.xlabel("Number of points in cloud")
plt.ylabel("Collision check time (ns)")
plt.title("Query performance of collision-checking structures")
plt.legend()
plt.show()


plt.plot(n_points, query_kdt_times, label="k-d tree (kiddo)")
plt.plot(n_points, query_forward_seq_times, label="forward tree, sequential")
plt.plot(n_points, query_forward_simd_times, label="forward tree, SIMD")
plt.plot(n_points, query_afftree_seq_times, label="affordance tree, sequential")
plt.plot(n_points, query_afftree_simd_times, label="affordance tree, SIMD")
plt.xlabel("Number of points in cloud")
plt.ylabel("Collision check time (ns)")
plt.title("Query performance of collision-checking structures")
plt.legend()
plt.show()

plt.plot(n_points, construct_kdt_times, label="k-d tree (kiddo)")
plt.plot(n_points, construct_forward_times, label="forward tree")
plt.plot(n_points, construct_afftree_times, label="affordance tree")
plt.xlabel("Number of points in cloud")
plt.ylabel("Tree construction time (ms)")
plt.title("Construction time of collision-checking structures")
plt.legend()
plt.show()