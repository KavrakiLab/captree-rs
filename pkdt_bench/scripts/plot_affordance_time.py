"""
USAGE: 
```
python ./plot_affordance_time.py path/to/perf.csv
```

Generate a plot of the runtime of the affordance tree construction and querying.
The first argument must point to a CSV file with the following columns:
1. Number of points in the tree
2. Time to construct the tree (in seconds)
3. Time for sequential queries (in seconds)
4. Time per point for batch queries (in seconds)
5. Time to build a conventional KD tree (in seconds)
6. Time to query a range in a conventional KD tree (in seconds)
"""

import csv
import matplotlib.pyplot as plt
import sys

csv_fname = sys.argv[1]
n_points = []
build_times = []
seq_times = []
simd_times = []
kdt_build_times = []
kdt_query_times = []
with open(csv_fname) as f:
    for row in csv.reader(f, delimiter=","):
        n_points.append(int(row[0]))
        build_times.append(float(row[1]))
        seq_times.append(float(row[2]))
        simd_times.append(float(row[3]))
        kdt_build_times.append(float(row[4]))
        kdt_query_times.append(float(row[5]))

plt.plot(n_points, build_times, label="Affordance tree")
plt.plot(n_points, kdt_build_times, label="KD tree")
plt.title("Construction time for trees")
plt.xlabel("Number of points in the tree")
plt.ylabel("Time for construction (s)")
plt.legend()
plt.show()


plt.plot(
    n_points, [p * 1e9 for p in seq_times], label="Sequential affordance tree queries"
)
plt.plot(n_points, [p * 1e9 for p in simd_times], label="SIMD affordance tree queries")
plt.plot(n_points, [p * 1e9 for p in kdt_query_times], label="KD tree queries")
plt.title("Nearest neighbors + collision check time")
plt.xlabel("Number of points in the tree")
plt.ylabel("Time for query and collision checking (ns)")
plt.legend()
plt.show()
