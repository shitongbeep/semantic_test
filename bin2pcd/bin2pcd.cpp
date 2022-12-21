#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

int main() {
    YAML::Node config = YAML::LoadFile("./config.yaml");
    // cout << config["pc_file"] << endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr points(
        new pcl::PointCloud<pcl::PointXYZI>);
    std::fstream input(config["pc_file"].as<std::string>(), std::ios::in | std::ios::binary);
    for (int i = 0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char*)&point.x, sizeof(float) * 3);
        input.read((char*)&point.intensity, sizeof(float));
        points->push_back(point);
    }
    input.close();

    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZI>(config["output_file"].as<std::string>(), *points, false);
}
