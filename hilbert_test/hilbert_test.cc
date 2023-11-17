#include <random>

#include <array>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include <highfive/H5File.hpp>
#include <hilbert-sort.h>
#include <pdqsort.h>

constexpr const std::size_t HilbertOrder = 1;
using Point = std::array<float, 3>;
using PointInt = std::array<uint16_t, 3>;

float dist(Point a, Point b)
{
    auto x = a[0] - b[0];
    auto y = a[1] - b[1];
    auto z = a[2] - b[2];
    return std::sqrt(x * x + y * y + z * z);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "No pointcloud file given. Using randomly generated points." << std::endl;
    }
    else
    {
        auto fname = std::string(argv[1]);
        std::cout << "Opening pointcloud file " << fname << "..." << std::endl;

        HighFive::File pointcloud_file(fname, HighFive::File::ReadOnly);
        auto dataset = pointcloud_file.getDataSet("pointcloud/points");
        auto raw_pointcloud = dataset.read<std::vector<Point>>();
        std::cout << raw_pointcloud.size() << " points loaded" << std::endl;

        double avg_pair_dist_before = 0;
        for (auto i = 0u; i < raw_pointcloud.size() - 1; ++i)
        {
            avg_pair_dist_before += dist(raw_pointcloud[i], raw_pointcloud[i + 1]);
        }
        avg_pair_dist_before /= raw_pointcloud.size() - 1;

        auto start_time = std::chrono::steady_clock::now();

        pdqsort(std::begin(raw_pointcloud), std::end(raw_pointcloud),
                [](const Point &a, const Point &b)
                {
                    const PointInt ai = {1000 * a[0], 1000 * a[1], 1000 * a[2]};
                    const PointInt bi = {1000 * b[0], 1000 * b[1], 1000 * b[2]};
                    auto const h1 = hilbert::hilbert_distance_by_coords<PointInt, HilbertOrder, 3>(ai);
                    auto const h2 = hilbert::hilbert_distance_by_coords<PointInt, HilbertOrder, 3>(bi);
                    return h1 < h2;
                });

        auto sort_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();

        double avg_pair_dist_after = 0;
        for (auto i = 0u; i < raw_pointcloud.size() - 1; ++i)
        {
            avg_pair_dist_after += dist(raw_pointcloud[i], raw_pointcloud[i + 1]);
        }
        avg_pair_dist_after /= raw_pointcloud.size() - 1;

        std::cout << "time (s): " << static_cast<double>(sort_time) / 1e9 << std::endl;
        std::cout << "avg seq-pair dist before: " << avg_pair_dist_before << std::endl;
        std::cout << "avg seq-pair dist after : " << avg_pair_dist_after << std::endl;
    }
}
