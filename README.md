# SIMD-Accelerated Collision Checking on Pointclouds

This is a collection of ideas for making collision checking between a robot and a pointcloud faster.

## Usage

Make sure to run `git submodule update --init --recursive`.

To run the benchmark, do `cargo run --release --bin bench path/to/my_pointcloud.hd5`.
To generate a CSV file with error distribution information, run
`cargo run --release --bin error path/to/my_pointcloud.hd5 > errors.csv`.
To parse this CSV file and generate plots, run
`python ./pkdt_bench/scripts/plot_error_hist.py errors.csv`.

## TODO

- Make crosswise table of parallel speedups by lane count and tree size
- Generalize benchmarking approach for arbitrary nearest-neighbors structures and test against the
  whole suite of bad implementations on [crates.io](crates.io)
- Build out an actual test suite that isn't just some numbers I made up
- Extend this approach to $k$-nearest-neighbors?
- Support exact nearest neighbors

## Relevant literature

- $k$-d forests: <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=834b4a31f33ccdbf4a92aa004031318df015f825>
- Progressive $k$-d trees: <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=834b4a31f33ccdbf4a92aa004031318df015f825>
- Application of non-tentative kd-tree search for image similarity: <https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=1240280&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2RvY3VtZW50LzEyNDAyODA=&tag=1>
- Point cloud downsampling: <https://arxiv.org/pdf/2307.02948.pdf>
- FLANN: <https://lear.inrialpes.fr/~douze/enseignement/2014-2015/presentation_papers/muja_flann.pdf>
- Discregrid: discrete generator for making SDFs: <https://github.com/InteractiveComputerGraphics/Discregrid/>
- Building a balanced $k$-d tree in $O(n \log n)$ (note: we tried this but it was slower in practice) <https://jcgt.org/published/0004/01/03/>
- comparison of ANN on a bunch of different things: <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8681160&casa_token=Sa3sfKdbiVUAAAAA:m51NJpk7aNiVvtLl9MFiapqL6_zvkYhM1aJ8UU8F1pJV8E3BVvDkuxVi_-IS7M9wHpzz_K5_&tag=1>
- k-NN optimizations for x86: <https://dl.acm.org/doi/pdf/10.1145/2807591.2807601>
- downsampling point clouds exactly: <https://arxiv.org/abs/2307.02948>, <https://github.com/koide3/caratheodory2/tree/main>
