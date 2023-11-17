# Benchmarks for nigh

Make sure to run `git submodule update --init --recursive` before building.

```sh
cmake -Bbuild -GNinja .
cmake --build build
./build/nigh_bench <file>
```
