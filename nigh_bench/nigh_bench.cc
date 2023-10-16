#include <random>

#include <array>
#include <iostream>
#include <limits>
#include <mutex>
#include <nigh/nigh_forward.hpp>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include <highfive/H5File.hpp>

#include <nigh/lp_space.hpp>
#include <nigh/kdtree_batch.hpp>

namespace nigh = unc::robotics::nigh;

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
        auto raw_pointcloud = dataset.read<std::vector<std::array<float, 3>>>();
        std::cout << raw_pointcloud.size() << " points loaded" << std::endl;

        using namespace unc::robotics::nigh;

        using Scalar = float;
        using Strategy = nigh::KDTreeBatch<8>;
        using Space = nigh::metric::L2Space<Scalar, 3>;
        using State = typename Space::Type;

        struct Node
        {
            State state_;
        };

        struct KeyFn
        {
            const State &operator()(const Node &n) const
            {
                return n.state_;
            }
        };

        Space space;
        Nigh<Node, Space, KeyFn, NoThreadSafety, Strategy> nn(space);

        auto start_time = std::chrono::steady_clock::now();
        for (const auto &point : raw_pointcloud)
        {
            nn.insert(Node{State(point.data())});
        }

        auto construction_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start_time)
                                     .count();

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<> dist(-2, 2);

        start_time = std::chrono::steady_clock::now();
        float total_d = 0;
        for (auto i = 0u; i < 1000000; ++i)
        {
            float x = dist(rng);
            float y = dist(rng);
            float z = dist(rng);

            auto on = nn.nearest(State{x, y, z});
            if (on)
            {
                auto [n, d] = *on;
                total_d += d;
            }
        }

        auto query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - start_time)
                              .count();

        std::cout << static_cast<double>(construction_time) / 1e9 << std::endl;
        std::cout << static_cast<double>(query_time) / 1e9 << std::endl;
    }
}
