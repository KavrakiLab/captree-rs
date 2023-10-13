#include <array>
#include <highfive/H5File.hpp>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Provide a pointcloud filename!\n";
    return -1;
  }

  auto filename = std::string(argv[1]);
  std::cout << "Reading " << filename << std::endl;
  HighFive::File pointcloud_file(filename, HighFive::File::ReadOnly);
  auto dataset = pointcloud_file.getDataSet("pointcloud/points");
  auto raw_pointcloud = dataset.read<std::vector<std::array<float, 3>>>();
  std::cout << raw_pointcloud.size() << " points loaded\n";
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  for (const auto &[x, y, z] : raw_pointcloud) {
    pointcloud->push_back(pcl::PointXYZ(x, y, z));
  }

  pcl::visualization::CloudViewer viewer("Pointcloud Viewer");
  viewer.showCloud(pointcloud);
  while (!viewer.wasStopped()) {
  }

  return 0;
}
