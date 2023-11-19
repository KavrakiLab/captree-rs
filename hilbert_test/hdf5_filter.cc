#include <chrono>
#include <cstdint>
#include <iostream>
#include <array>
#include <filesystem>

#include <highfive/H5File.hpp>
#include <pdqsort.h>
#include <fmt/core.h>

#include "cxxopts.hpp"

#include "hilbert.hh"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>

namespace co = cxxopts;
namespace fs = std::filesystem;

using Point = std::array<float, 3>;

constexpr const uint_fast32_t HILBERT_FACTOR = 1000;

inline constexpr auto remap_point(float x, float min, float max) -> uint_fast32_t
{
    return ((x - min) / (max - min)) * static_cast<float>(HILBERT_FACTOR);
}

struct Timer
{
    Timer()
    {
        start = std::chrono::steady_clock::now();
    }

    auto click()
    {
        duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
                .count();
    }

    std::chrono::time_point<std::chrono::steady_clock> start;
    std::size_t duration;
};

inline constexpr auto sql2(Point a, Point b) -> float
{
    const auto x = a[0] - b[0];
    const auto y = a[1] - b[1];
    const auto z = a[2] - b[2];

    return x * x + y * y + z * z;
}

inline constexpr auto l2(Point a, Point b) -> float
{
    return std::sqrt(sql2(a, b));
}

auto load_pc(fs::path filepath) -> std::vector<Point>
{
    HighFive::File pointcloud_file(filepath, HighFive::File::ReadOnly);
    auto dataset = pointcloud_file.getDataSet("pointcloud/points");
    return dataset.read<std::vector<Point>>();
}

auto save_pc(const std::vector<Point> &pc, fs::path filepath)
{
    HighFive::File export_file(filepath, HighFive::File::Truncate);
    export_file.createDataSet("pointcloud/points", pc);
}

template <uint_fast32_t b>
struct Halton
{
    auto next() -> float
    {
        auto x = d - n;
        if (x == 1)
        {
            n = 1;
            d *= b;
        }
        else
        {
            auto y = d / b;
            while (x <= y)
            {
                y /= b;
            }

            n = (b + 1) * y - x;
        }

        return static_cast<float>(n) / static_cast<float>(d);
    }

    uint_fast32_t n{0};
    uint_fast32_t d{1};
};

auto process(const std::vector<Point> &pc, float min_dist, float max_range, Point origin,
             float pre_halton_threshold, float post_halton_threshold) -> std::vector<Point>
{
    const auto sqdist = min_dist * min_dist;
    const auto sqrange = max_range * max_range;
    const auto min = std::min({origin[0] - max_range, origin[1] - max_range, origin[2] - max_range});
    const auto max = std::min({origin[0] + max_range, origin[1] + max_range, origin[2] + max_range});

    std::vector<std::pair<Point, uint_fast32_t>> hilbert;
    hilbert.reserve(pc.size());

    Halton<3> pre_halton;

    for (auto i = 0u; i < pc.size(); ++i)
    {
        auto &&p = pc[i];
        if (not(pre_halton_threshold > 0.0 and pre_halton.next() < pre_halton_threshold) and
            sql2(p, origin) < sqrange)
        {
            auto x = remap_point(p[0], min, max);
            auto y = remap_point(p[1], min, max);
            auto z = remap_point(p[2], min, max);

            hilbert.emplace_back(p, hilbert_reimp_pdep(x, y, z));
        }
    }

    pdqsort_branchless(std::begin(hilbert), std::end(hilbert),
                       [](auto &&a, auto &&b) { return a.second < b.second; });

    std::vector<Point> filtered;
    filtered.reserve(hilbert.size());
    filtered.emplace_back(hilbert[0].first);

    Halton<3> post_halton;

    for (auto i = 1u; i < hilbert.size(); ++i)
    {
        auto &&p = hilbert[i].first;
        if (not(post_halton_threshold > 0.0 and post_halton.next() < post_halton_threshold) and
            sql2(p, filtered.back()) > sqdist)
        {
            filtered.emplace_back(p);
        }
    }

    return filtered;
}

inline constexpr auto average_adjacent_distance(const std::vector<Point> &pc) -> float
{
    float average_distance = 0;
    for (auto i = 1u; i < pc.size(); ++i)
    {
        average_distance += l2(pc[i - 1], pc[i]);
    }

    return average_distance / (pc.size() - 1);
}

int main(int argc, char **argv)
{
    co::Options options("hdf5_filter", "Filter a pointcloud stored in a hdf5 file.");

    options.add_options()                                                                                //
        ("f,input_file", "Filename of hdf5 file", co::value<std::string>())                              //
        ("o,output_file", "Output filename", co::value<std::string>()->default_value("filtered.hdf5"))   //
        ("m,minimum_distance", "Min. dist. between points", co::value<float>()->default_value("0.01"))   //
        ("r,maximum_range", "Max. dist. of points to origin", co::value<float>()->default_value("1.2"))  //
        ("x,origin_x", "x-value of origin", co::value<float>()->default_value("0.0"))                    //
        ("y,origin_y", "y-value of origin", co::value<float>()->default_value("0.0"))                    //
        ("z,origin_z", "z-value of origin", co::value<float>()->default_value("0.0"))                    //
        ("s,halton_pre", "Pre-sort Halton threshold", co::value<float>()->default_value("0.0"))          //
        ("p,halton_post", "Post-sort Halton threshold", co::value<float>()->default_value("0.0"))        //
        ("h,help", "Print usage")                                                                        //
        ;

    options.parse_positional({"input_file"});
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (not result.count("input_file"))
    {
        std::cout << "No file provided!" << std::endl;
        std::cout << options.help() << std::endl;
        return 0;
    }

    fs::path filepath = result["input_file"].as<std::string>();

    if (not fs::exists(filepath))
    {
        std::cout << "File does not exist!" << std::endl;
        std::cout << options.help() << std::endl;
        return 0;
    }

    Timer load_timer;
    auto pc = load_pc(filepath);
    load_timer.click();

    auto start_avg_dist = average_adjacent_distance(pc);
    auto start_n = pc.size();

    Timer process_timer;
    auto filtered_pc =
        process(pc, result["minimum_distance"].as<float>(), result["maximum_range"].as<float>(),
                Point{
                    result["origin_x"].as<float>(),
                    result["origin_y"].as<float>(),
                    result["origin_z"].as<float>(),
                },
                result["halton_pre"].as<float>(), result["halton_post"].as<float>());
    process_timer.click();

    auto final_avg_dist = average_adjacent_distance(filtered_pc);
    auto final_n = filtered_pc.size();

    fmt::print("Loading pointcloud took {0:0.2f}ms\n", load_timer.duration / 1e6);
    fmt::print("Processing pointcloud took {0:0.2f}ms\n", process_timer.duration / 1e6);
    fmt::print("Start avg. adj. dist: {0:0.5f} / {1:6d} points\n", start_avg_dist, start_n);
    fmt::print("Final avg. adj. dist: {0:0.5f} / {1:6d} points\n", final_avg_dist, final_n);

    save_pc(filtered_pc, fs::path(result["output_file"].as<std::string>()));

    return 0;
}
