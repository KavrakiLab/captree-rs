import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    radii = np.genfromtxt(sys.argv[1], usecols=0, delimiter=',')
    ns = np.genfromtxt(sys.argv[1], usecols=(1, 2, 3, 4, 5, 6), delimiter=',')
    for i, col in enumerate(ns.T):
        plt.plot(radii * 100, col, label=f"{i + 1} permutations")
    plt.xlabel("Filter radius (cm)")
    plt.ylabel("Number of points after filtering")
    plt.semilogy()
    plt.title("Effectiveness of space-filling curve filter by radius")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
