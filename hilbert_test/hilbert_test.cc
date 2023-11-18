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
#include <pdqsort.h>
#include "hilbert-sort.h"

constexpr const std::size_t Dim = 3;
constexpr const std::size_t HilbertOrder = 1;
constexpr const std::size_t HilbertFactor = 1000;
using Point = std::array<float, Dim>;
using PointInt = std::array<uint_fast32_t, Dim>;

constexpr auto remap_point(float x, float min, float max) -> uint32_t
{
    return ((x - min) / (max - min)) * static_cast<float>(HilbertFactor);
}

struct PointHilbert
{
    PointHilbert(Point a) : point(a){};

    constexpr auto compute(float min, float max)
    {
        const PointInt pi = {remap_point(point[0], min, max), remap_point(point[1], min, max),
                             remap_point(point[2], min, max)};
        hilbert = hilbert::hilbert_distance_by_coords<PointInt, HilbertOrder, Dim>(pi);
    }

    std::array<float, Dim> point;
    uint32_t hilbert;
};

auto dist(Point a, Point b) -> float
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
        std::cout << "No pointcloud file given." << std::endl;
        return 0;
    }

    float filter_range = 1e-2;
    if (argc > 2)
    {
        filter_range = stof(argv[2]);
    }

    float max_distance = 1.2;
    if (argc > 3)
    {
        max_distance = stof(argv[2]);
    }

    std::cout << "Filter range " << filter_range << std::endl;

    auto fname = std::string(argv[1]);
    std::cout << "Opening pointcloud file " << fname << "..." << std::endl;

    HighFive::File pointcloud_file(fname, HighFive::File::ReadOnly);
    auto dataset = pointcloud_file.getDataSet("pointcloud/points");
    auto raw_pointcloud = dataset.read<std::vector<Point>>();
    std::cout << raw_pointcloud.size() << " points loaded" << std::endl;

    float min = 0, max = 0;
    double avg_pair_dist_before = 0;
    for (auto i = 0u; i < raw_pointcloud.size() - 1; ++i)
    {
        avg_pair_dist_before += dist(raw_pointcloud[i], raw_pointcloud[i + 1]);

        min = std::min({min, raw_pointcloud[i][0], raw_pointcloud[i][1], raw_pointcloud[i][2]});
        max = std::max({max, raw_pointcloud[i][0], raw_pointcloud[i][1], raw_pointcloud[i][2]});
    }
    avg_pair_dist_before /= raw_pointcloud.size() - 1;

    // std::vector<PointHilbert> hilbert_points(raw_pointcloud.begin(), raw_pointcloud.end());
    std::vector<PointHilbert> hilbert_points;
    hilbert_points.reserve(raw_pointcloud.size());

    auto start_time = std::chrono::steady_clock::now();

    for (auto i = 0u; i < raw_pointcloud.size(); ++i)
    {
        if (dist(raw_pointcloud[i], {0, 0, 0}) < max_distance)
        {
            hilbert_points.emplace_back(raw_pointcloud[i]);
        }
    }

    for (auto i = 0u; i < hilbert_points.size(); ++i)
    {
        hilbert_points[i].compute(min, max);
    }

    pdqsort_branchless(std::begin(hilbert_points), std::end(hilbert_points),
                       [](const PointHilbert &a, const PointHilbert &b) { return a.hilbert < b.hilbert; });

    auto sort_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_time)
            .count();

    double avg_pair_dist_after = 0;
    for (auto i = 0u; i < hilbert_points.size() - 1; ++i)
    {
        avg_pair_dist_after += dist(hilbert_points[i].point, hilbert_points[i + 1].point);
    }
    avg_pair_dist_after /= raw_pointcloud.size() - 1;

    std::cout << "time (s): " << static_cast<double>(sort_time) / 1e9 << std::endl;
    std::cout << "avg seq-pair dist before: " << avg_pair_dist_before << std::endl;
    std::cout << "avg seq-pair dist after : " << avg_pair_dist_after << std::endl;

    std::vector<Point> filtered;
    filtered.resize(raw_pointcloud.size());
    filtered[0] = hilbert_points[0].point;

    auto j = 0u;
    for (auto i = 1u; i < hilbert_points.size(); ++i)
    {
        const float d = dist(filtered[j], hilbert_points[i].point);
        if (d > filter_range)
        {
            filtered[++j] = hilbert_points[i].point;
        }
    }

    filtered.resize(j);
    std::cout << "filtered size: " << j << std::endl;

    double avg_pair_dist_filtered = 0;
    for (auto i = 0u; i < filtered.size() - 1; ++i)
    {
        avg_pair_dist_filtered += dist(filtered[i], filtered[i + 1]);
    }
    avg_pair_dist_filtered /= raw_pointcloud.size() - 1;
    std::cout << "avg seq-pair dist filtered: " << avg_pair_dist_filtered << std::endl;

    HighFive::File export_file("filtered.hdf5", HighFive::File::Truncate);
    export_file.createDataSet("pointcloud/points", filtered);
}
