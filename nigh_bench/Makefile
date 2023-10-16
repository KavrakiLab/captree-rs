CC=clang
CXX=clang++

all: nigh_bench

clean:
	rm -rf nigh_bench

nigh_bench: nigh_bench.cpp
	${CXX} -O3 `pkg-config --cflags eigen3` ./nigh_bench.cpp -o nigh_bench