#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <string>

using namespace std;

void PointCloudXYZtoXYZview(
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz,
    pcl::PointCloud<pcl::PointWithViewpoint>::Ptr xyzview) {
    pcl::copyPointCloud(*xyz, *xyzview);
    for (auto &point : xyzview->points) {
        point.vp_x =
            sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        point.vp_y = asinf(point.z / point.vp_x) / M_PI * 180.;
        point.vp_z = atan2f(point.y, point.x) / M_PI * 180.;
    }
}

void RemoveOutlier(pcl::PointCloud<pcl::PointXYZ>::Ptr xyz,
                   pcl::PointCloud<pcl::PointWithViewpoint>::Ptr xyzview,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr output_xyz,
                   int innerpoints, float range_ratio, float fai_step,
                   float theta_step) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(xyz);
    for (int pts_idx = 0; pts_idx < xyz->points.size(); pts_idx++) {
        float range_step = range_ratio * xyzview->points[pts_idx].vp_x;
        vector<int> pointIdxNKNSearch(innerpoints);
        vector<float> pointRadiusSquaredDistance;
        kdtree.nearestKSearch(xyz->points[pts_idx], innerpoints,
                              pointIdxNKNSearch, pointRadiusSquaredDistance);
        bool goodpoint = true;
        for (int i = 0; i < innerpoints; i++) {
            if (abs(xyzview->points[pts_idx].vp_x -
                    xyzview->points[pointIdxNKNSearch[i]].vp_x) >
                    range_step / 2 ||
                abs(xyzview->points[pts_idx].vp_y -
                    xyzview->points[pointIdxNKNSearch[i]].vp_y) >
                    fai_step / 2 ||
                min((360 - abs(xyzview->points[pts_idx].vp_z -
                                xyzview->points[pointIdxNKNSearch[i]].vp_z)),
                    abs(xyzview->points[pts_idx].vp_z -
                        xyzview->points[pointIdxNKNSearch[i]].vp_z)) >
                    theta_step / 2) {
                goodpoint = false;
                break;
            }
        }
        if (goodpoint) {
            output_xyz->points.push_back(xyz->points[pts_idx]);
        }
    }
}

int main() {
    YAML::Node config = YAML::LoadFile("./config.yaml");
    string filter_input = config["filter_pcd"].as<string>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PCDReader reader;
    reader.read(filter_input, *input_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::copyPointCloud(*input_cloud, *input_xyz);
    pcl::visualization::PCLVisualizer viewer("filter");
    // viewer.addPointCloud(input_xyz);
    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);

    //* voxel pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_xyz(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> vol;
    vol.setInputCloud(input_xyz);
    vol.setLeafSize(0.05f, 0.05f, 0.05f);
    vol.filter(*voxel_xyz);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        voxel_xyz_rgb(voxel_xyz, 255, 0, 0);
    // viewer.addPointCloud(voxel_xyz, voxel_xyz_rgb, "voxel_xyz");

    //* remove outlier
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeoutlier_xyz(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filt;
    filt.setInputCloud(input_xyz);
    filt.setMeanK(50);
    filt.setStddevMulThresh(0.1);
    filt.filter(*removeoutlier_xyz);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        removeoutlier_xyz_rgb(voxel_xyz, 0, 0, 255);
    // viewer.addPointCloud(removeoutlier_xyz, removeoutlier_xyz_rgb,
    // "removeoutlier_xyz");

    //* radius remove outlier
    pcl::PointCloud<pcl::PointXYZ>::Ptr removeradius_xyz(
        new pcl::PointCloud<pcl::PointXYZ>());
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius;
    radius.setInputCloud(input_xyz);
    radius.setRadiusSearch(0.1);
    radius.setMinNeighborsInRadius(3);
    radius.setKeepOrganized(true);
    radius.filter(*removeradius_xyz);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        removeradius_xyz_rgb(voxel_xyz, 0, 255, 0);
    // viewer.addPointCloud(removeradius_xyz, removeradius_xyz_rgb,
    //                      "removeradius_xyz");

    pcl::PointCloud<pcl::PointWithViewpoint>::Ptr point_view(
        new pcl::PointCloud<pcl::PointWithViewpoint>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr my_remove_outlier(
        new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudXYZtoXYZview(input_xyz, point_view);
    RemoveOutlier(input_xyz, point_view, my_remove_outlier, 5, 0.01, 0.4,
                  360. / 2048. * 4.);
    cout << my_remove_outlier->points.size();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        my_remove_outlier_rgb(my_remove_outlier, 0, 255, 255);
    viewer.addPointCloud(my_remove_outlier, my_remove_outlier_rgb,
                         "my_remove_outlier");

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}