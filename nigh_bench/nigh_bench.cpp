
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

#include "nigh/src/nigh/kdtree_batch.hpp"
#include "nigh/src/nigh/metric/lp.hpp"

namespace nigh = unc::robotics::nigh;

using State = Eigen::Vector3d;
using Metric = nigh::metric::LP<2>;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "No pointcloud file given. Using randomly generated points." << std::endl;
    } else {
        auto fname = std::string(argv[1]);
        std::cout << "Opening pointcloud file " << fname << "..." << std::endl;

        HighFive::File pointcloud_file(fname, HighFive::File::ReadOnly);
        auto dataset = pointcloud_file.getDataSet("pointcloud/points");
        auto raw_pointcloud = dataset.read<std::vector<std::array<float, 3>>>();
        std::cout << raw_pointcloud.size() << " points loaded" << std::endl;
    }
}