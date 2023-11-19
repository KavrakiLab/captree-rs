#include <array>
#include <highfive/H5File.hpp>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/impl/point_types.hpp>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

struct Entry
{
    float s;
    float r;
    float g;
    float b;
};

static const Entry VIRIDIS[32] = {  //
    {0.0, 0.2823645529290169, 0.0, 0.3310101940118055},
    {0.03225806451612903, 0.29722632020441175, 0.027183413963800338, 0.37864029621447814},
    {0.06451612903225806, 0.3072803803150455, 0.0802447375229367, 0.4212673579883743},
    {0.0967741935483871, 0.3123264966480157, 0.12584534969096695, 0.457985073737091},
    {0.12903225806451613, 0.31237425118765544, 0.16879778927302974, 0.48815673654998043},
    {0.16129032258064516, 0.3077875785131497, 0.2101981170027718, 0.511619450888943},
    {0.1935483870967742, 0.29933085741427456, 0.25010033561427364, 0.5288208146468127},
    {0.22580645161290322, 0.2880586895294369, 0.28835583248860613, 0.5407177619131058},
    {0.25806451612903225, 0.2750965242322183, 0.3248913223445997, 0.5485120229082456},
    {0.29032258064516125, 0.2614262440352356, 0.3597968855678623, 0.5533850614666772},
    {0.3225806451612903, 0.2477338952312153, 0.39328759309143785, 0.5563075495959575},
    {0.3548387096774194, 0.23426522372462222, 0.4256765238335555, 0.5579283364248078},
    {0.3870967741935484, 0.2208774366275068, 0.4572924484791127, 0.5585488414824472},
    {0.4193548387096774, 0.2071151706305011, 0.48844291588547933, 0.5581501761803592},
    {0.45161290322580644, 0.19238993131530202, 0.5193782701027633, 0.5564329716762256},
    {0.4838709677419355, 0.17626507220153706, 0.5502647724624314, 0.5529002630158235},
    {0.5161290322580645, 0.1589045039992837, 0.5811708881961466, 0.5469358434427122},
    {0.5483870967741935, 0.14180310141349906, 0.6120609032437101, 0.537882137387205},
    {0.5806451612903225, 0.1289290768789858, 0.6428125092789465, 0.5251057926660444},
    {0.6129032258064516, 0.12737844015895444, 0.6732171143800528, 0.5080488847567046},
    {0.6451612903225806, 0.1441804301921067, 0.7030136844759319, 0.4862684965591734},
    {0.6774193548387096, 0.1805504147795526, 0.7318951275563422, 0.4594108399264241},
    {0.7096774193548387, 0.23246629839421223, 0.7595227413720251, 0.4272148754878393},
    {0.7419354838709677, 0.2955966115638969, 0.7855342571397639, 0.3894744537385309},
    {0.7741935483870968, 0.36690903818194603, 0.8095679467648405, 0.3460474232378068},
    {0.8064516129032258, 0.44452103469874904, 0.8312792890501436, 0.29679122529015456},
    {0.8387096774193548, 0.5270107870979416, 0.8503789777911802, 0.24157408056415985},
    {0.8709677419354839, 0.6129677032098788, 0.8667148025192438, 0.18037880054649338},
    {0.9032258064516129, 0.7006476041657969, 0.8804041088681586, 0.114029402487933},
    {0.9354838709677419, 0.7879194958353214, 0.8919655007329689, 0.050317962262272926},
    {0.967741935483871, 0.8726031817793708, 0.9023098719403728, 0.043816043656816815},
    {1.0, 0.9529994532916154, 0.9125452328290099, 0.11085876909361342}};

float remap(float a1, float a2, float av, float b1, float b2)
{
    float dat = fabs(a2 - a1);
    float dav = fabs(av - a1);

    float dbt = fabs(b2 - b1);
    float dbv = dbt * (dav / dat);

    float bv = b1 + dbv;

    return bv;
}

void colormap(float s, float &r, float &g, float &b, unsigned int size, const Entry map[])
{
    for (unsigned int i = 1; i < size; ++i)
    {
        const Entry &p = map[i - 1];
        const Entry &n = map[i];

        if (s < n.s)
        {
            r = remap(p.s, n.s, s, p.r, n.r);
            g = remap(p.s, n.s, s, p.g, n.g);
            b = remap(p.s, n.s, s, p.b, n.b);
            return;
        }
    }

    r = map[size - 1].r;
    g = map[size - 1].g;
    b = map[size - 1].b;
}

inline constexpr auto l2(float x, float y, float z) -> float
{
    return std::sqrt(x * x + y * y + z * z);
}

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Provide a pointcloud filename!\n";
        return -1;
    }

    auto filename = std::string(argv[1]);
    std::cout << "Reading " << filename << std::endl;
    HighFive::File pointcloud_file(filename, HighFive::File::ReadOnly);
    auto dataset = pointcloud_file.getDataSet("pointcloud/points");
    auto raw_pointcloud = dataset.read<std::vector<std::array<float, 3>>>();
    std::cout << raw_pointcloud.size() << " points loaded\n";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    float max = 0;
    for (const auto &[x, y, z] : raw_pointcloud)
    {
        max = std::max({l2(x, y, z), max});
    }

    for (const auto &[x, y, z] : raw_pointcloud)
    {
        float r, g, b;
        float d = l2(x, y, z);
        colormap(d / max, r, g, b, 32, VIRIDIS);
        pointcloud->push_back(pcl::PointXYZRGB(x, y, z, r * 255, g* 255, b * 255));
    }

    pcl::visualization::PCLVisualizer::Ptr viewer( //
        new pcl::visualization::PCLVisualizer("Pointcloud Viewer"));

    viewer->setBackgroundColor(0.2, 0.2, 0.2);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pointcloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(pointcloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(16);
        std::this_thread::sleep_for(16ms);
    }

    return 0;
}
