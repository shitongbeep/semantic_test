#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <string>

using namespace std;

void bin2pcd(const char* bin_path, const char* pcd_path);
void remove_outlier(const char* velodyne_path, const char* denser_path,
                    const char* denser_refine_path);

int main() {
    YAML::Node config = YAML::LoadFile("./config.yaml");
    string view_path = config["view_path"].as<string>();

    string denser_bin_path = view_path + "denser/";
    string velodyne_bin_path = view_path + "velodyne/";
    string denser_pcd_path = view_path + "denser_pcd/";
    string velodyne_pcd_path = view_path + "velodyne_pcd/";
    if (access(denser_pcd_path.c_str(), F_OK))
        bin2pcd(denser_bin_path.c_str(), denser_pcd_path.c_str());
    if (access(velodyne_pcd_path.c_str(), F_OK))
        bin2pcd(velodyne_bin_path.c_str(), velodyne_pcd_path.c_str());

    string denser_refined_path = view_path + "denser_refined/";
    if (access(denser_refined_path.c_str(), F_OK))
        remove_outlier(velodyne_pcd_path.c_str(), denser_pcd_path.c_str(),
                       denser_refined_path.c_str());

    DIR* dir;
    dirent* entry;
    dir = opendir(velodyne_pcd_path.c_str());
    vector<string> velodyne_file_list;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        velodyne_file_list.push_back(velodyne_pcd_path + string(entry->d_name));
    }
    sort(velodyne_file_list.begin(), velodyne_file_list.end());
    dir = opendir(denser_refined_path.c_str());
    vector<string> refined_file_list;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        refined_file_list.push_back(denser_refined_path +
                                    string(entry->d_name));
    }
    sort(refined_file_list.begin(), refined_file_list.end());

    pcl::visualization::PCLVisualizer viewer("visulize");
    // viewer.addPointCloud(input_xyz);
    int view_port_1 = 1;
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, view_port_1);
    viewer.addCoordinateSystem(1.0, "cloud1", view_port_1);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, view_port_1);

    int view_port_2 = 2;
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, view_port_2);
    viewer.addCoordinateSystem(1.0, "cloud2", view_port_2);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, view_port_2);

    pcl::PCDReader reader;
    for (int pc_idx = 0; pc_idx < refined_file_list.size(); pc_idx++) {
        viewer.removeAllPointClouds(view_port_1);
        viewer.removeAllPointClouds(view_port_2);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr refined_xyzrgb(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        reader.read(refined_file_list[pc_idx], *refined_xyzrgb);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr refined_xyz(
        //     new pcl::PointCloud<pcl::PointXYZ>());
        // pcl::copyPointCloud(*refined_xyzi, *refined_xyz);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr velodyne_xyzrgb(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        reader.read(velodyne_file_list[pc_idx], *velodyne_xyzrgb);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr velodyne_xyz(
        //     new pcl::PointCloud<pcl::PointXYZ>());
        // pcl::copyPointCloud(*velodyne_xyzi, *velodyne_xyz);

        viewer.addPointCloud(velodyne_xyzrgb, "velodyne_xyzrgb", view_port_1);
        viewer.addPointCloud(refined_xyzrgb, "refined_xyzrgb", view_port_2);

        viewer.spinOnce(100);
    }

    return 0;
}

vector<uint8_t> jet_colormap(int d) {
    if (d > 255 || d < 0) return vector<uint8_t>{0, 0, 0};
    vector<int> ret_int = {0, 0, 0};
    if (d < 32)
        ret_int = {128 + 4 * d, 0, 0};
    else if (d == 32)
        ret_int = {255, 0, 0};
    else if (d < 96)
        ret_int = {255, 4 + 4 * (d - 33), 0};
    else if (d == 96)
        ret_int = {254, 255, 2};
    else if (d < 159)
        ret_int = {250 - 4 * (d - 97), 255, 6 + 4 * (d - 97)};
    else if (d == 159)
        ret_int = {1, 255, 254};
    else if (d < 224)
        ret_int = {0, 252 - 4 * (d - 160), 255};
    else
        ret_int = {0, 0, 252 - 4 * (d - 224)};
    // cout << ret_int[0] << ret_int[1] << ret_int[2] << endl;
    return vector<uint8_t>{boost::numeric_cast<uint8_t>(ret_int[0]),
                           boost::numeric_cast<uint8_t>(ret_int[1]),
                           boost::numeric_cast<uint8_t>(ret_int[2])};
}

void bin2pcd(const char* bin_path, const char* pcd_path) {
    mkdir(pcd_path, 0777);
    DIR* dir;
    dirent* entry;
    dir = opendir(bin_path);
    //* read bin file abs path to a list
    vector<string> bin_file_list;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        bin_file_list.push_back(string(bin_path) + string(entry->d_name));
    }
    sort(bin_file_list.begin(), bin_file_list.end());
    //* convert every bin file to pcd file
    pcl::PCDWriter writer;
    for (int file_idx = 0; file_idx < bin_file_list.size(); file_idx++) {
        cout << "\r";
        //* get output file abs path
        string output_file = string(pcd_path) +
                             bin_file_list[file_idx].substr(
                                 bin_file_list[file_idx].size() - 10, 7) +
                             "pcd";

        //* read bin file and convert to pcd
        string bin_file = bin_file_list[file_idx];
        pcl::PointCloud<pcl::PointXYZI>::Ptr points(
            new pcl::PointCloud<pcl::PointXYZI>);
        std::fstream input(bin_file, std::ios::in | std::ios::binary);
        float max_dist = 0.;
        for (int i = 0; input.good() && !input.eof(); i++) {
            pcl::PointXYZI point;
            input.read((char*)&point.x, sizeof(float) * 3);
            input.read((char*)&point.intensity, sizeof(float));
            float cur_dist = sqrt(point.x * point.x + point.y * point.y);
            if (cur_dist > max_dist) max_dist = cur_dist;
            points->push_back(point);
        }
        input.close();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*points, *xyzrgb);
        for (auto& point : xyzrgb->points) {
            float cur_dist = sqrt(point.x * point.x + point.y * point.y);
            float cur_d = cur_dist / max_dist * 255;
            // cout << cur_d << ' ' << int(cur_d) << endl;
            vector<uint8_t> rgb = jet_colormap(cur_d);
            point.b = rgb[0];
            point.g = rgb[1];
            point.r = rgb[2];
            // cout << point;
        }

        writer.write<pcl::PointXYZRGB>(output_file, *xyzrgb, false);
        cout << file_idx << '/' << bin_file_list.size();
    }
}

static void RemoveOutlier(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_xyzrgb,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_xyzrgb,
                          int innerpoints, float range_ratio, float fai_step,
                          float theta_step) {
    //* generate polor coordinate
    pcl::PointCloud<pcl::PointWithViewpoint>::Ptr xyzpolor(
        new pcl::PointCloud<pcl::PointWithViewpoint>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(*input_xyzrgb, *input_xyz);
    pcl::copyPointCloud(*input_xyz, *xyzpolor);
    for (auto& point : xyzpolor->points) {
        point.vp_x =
            sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        point.vp_y = asinf(point.z / point.vp_x) / M_PI * 180.;
        point.vp_z = atan2f(point.y, point.x) / M_PI * 180.;
    }
    //*
    // pcl::PointCloud<pcl::PointXYZ>::Ptr output_xyz(
    //     new pcl::PointCloud<pcl::PointXYZ>());
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(input_xyz);
    for (int pts_idx = 0; pts_idx < input_xyz->points.size(); pts_idx++) {
        float range_step = range_ratio * xyzpolor->points[pts_idx].vp_x;
        vector<int> pointIdxNKNSearch(innerpoints);
        vector<float> pointRadiusSquaredDistance;
        kdtree.nearestKSearch(input_xyz->points[pts_idx], innerpoints,
                              pointIdxNKNSearch, pointRadiusSquaredDistance);
        bool goodpoint = true;
        for (int i = 0; i < innerpoints; i++) {
            if (abs(xyzpolor->points[pts_idx].vp_x -
                    xyzpolor->points[pointIdxNKNSearch[i]].vp_x) >
                    range_step / 2 ||
                abs(xyzpolor->points[pts_idx].vp_y -
                    xyzpolor->points[pointIdxNKNSearch[i]].vp_y) >
                    fai_step / 2 ||
                min((360 - abs(xyzpolor->points[pts_idx].vp_z -
                               xyzpolor->points[pointIdxNKNSearch[i]].vp_z)),
                    abs(xyzpolor->points[pts_idx].vp_z -
                        xyzpolor->points[pointIdxNKNSearch[i]].vp_z)) >
                    theta_step / 2) {
                goodpoint = false;
                break;
            }
        }
        if (goodpoint) {
            output_xyzrgb->points.push_back(input_xyzrgb->points[pts_idx]);
        }
    }
}

void remove_outlier(const char* velodyne_path, const char* denser_path,
                    const char* denser_refined_path) {
    mkdir(denser_refined_path, 0777);
    vector<string> velodyne_pcd_list;
    vector<string> denser_pcd_list;
    // vector<string> redined_pcd_list;
    DIR* dir;
    dirent* entry;
    dir = opendir(velodyne_path);
    //* read velodyne file abs path to a list
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        velodyne_pcd_list.push_back(string(velodyne_path) +
                                    string(entry->d_name));
    }
    sort(velodyne_pcd_list.begin(), velodyne_pcd_list.end());
    dir = opendir(denser_path);
    //* read unrefined dense pointcloud file abs path to a list
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        denser_pcd_list.push_back(string(denser_path) + string(entry->d_name));
    }
    sort(denser_pcd_list.begin(), denser_pcd_list.end());
    //* refine every dense pointcloud and merge with velodyne piontcloud
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    for (int pc_idx = 0; pc_idx < denser_pcd_list.size(); pc_idx++) {
        cout << "\r" << pc_idx << '/' << denser_pcd_list.size();
        //* remove outlier
        string denser_pcd_file = denser_pcd_list[pc_idx];
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr denser_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        reader.read(denser_pcd_file, *denser_cloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr refine_dense_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        RemoveOutlier(denser_cloud, refine_dense_cloud, 5, 0.01, 0.4,
                      360. / 2048. * 4.);
        //* merge with velodyne
        string velodyne_pcd_file = velodyne_pcd_list[pc_idx];
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr velodyne_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        reader.read(velodyne_pcd_file, *velodyne_cloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr refined_point_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>());
        *refined_point_cloud = *velodyne_cloud + *refine_dense_cloud;
        //* write to pcd file
        string denser_refined_output_path =
            string(denser_refined_path) +
            denser_pcd_file.substr(denser_pcd_file.size() - 10, 10);
        writer.write<pcl::PointXYZRGB>(denser_refined_output_path,
                                       *refined_point_cloud, false);
    }
}
