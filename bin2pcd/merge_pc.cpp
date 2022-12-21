#include <dirent.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;

using namespace Eigen;

int main() {
    YAML::Node config = YAML::LoadFile("./config.yaml");
    // cout << config["pc_file"] << endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr points(
        new pcl::PointCloud<pcl::PointXYZI>);
    string kitti_path = config["semantickitti_path"].as<string>();
    YAML::Node sequence_node = config["merge_sequence"];
    int sequence_list[22];
    memset(sequence_list, -1, 22);
    for (std::size_t i = 0; i < sequence_node.size(); i++) {
        sequence_list[i] = sequence_node[i].as<int>();
        // cout<<sequence_list[i];
    }
    for (int seq = 0; sequence_list[seq] != -1 && seq < 22; seq++) {
        cout << "start sequence: " << seq << endl;
        //* 获得文件路径
        std::stringstream ss;
        ss << std::setw(2) << std::setfill('0') << sequence_list[seq];
        string sequence_idx;
        ss >> sequence_idx;
        string cur_data_path = kitti_path + sequence_idx + '/';
        string velodyne_folder =
            cur_data_path + config["velodyne_folder"].as<string>();
        string poses_file = cur_data_path + config["poses_file"].as<string>();
        string calib_file = cur_data_path + config["calib_file"].as<string>();
        string output_path = cur_data_path +
                             config["merge_output_folder"].as<string>() +
                             "mergepcd.pcd";
        //* 读取pose
        std::vector<Matrix4f> T_w_cam;
        std::fstream read_poses_file(poses_file);
        while (!read_poses_file.eof()) {
            Eigen::Matrix4f cur_pose = Matrix4f::Identity();
            for (int i = 0; i < 12; i++) {
                read_poses_file >> cur_pose(i / 4, i % 4);
            }
            T_w_cam.push_back(cur_pose);
        }
        Matrix4f T_cam0_w = T_w_cam[0].inverse();
        //* 读取变换矩阵
        std::fstream read_calib_file(calib_file);
        Eigen::Matrix4f T_cam0_velo = Matrix4f::Identity();
        for (int i = 0; i < 4; i++) {
            char P[512];
            read_calib_file.getline(P, 512);
        }
        char rubbish[512];
        read_calib_file >> rubbish;
        for (int i = 0; i < 12; i++) {
            read_calib_file >> T_cam0_velo(i / 4, i % 4);
        }
        Matrix4f T_velo_cam0 = T_cam0_velo.inverse();
        //* 依次读取点云路径
        DIR* dir;
        dirent* entry;
        dir = opendir(velodyne_folder.c_str());
        std::vector<string> velodyne_pc_file;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_name[0] == '.') continue;
            velodyne_pc_file.push_back(velodyne_folder + string(entry->d_name));
        }
        std::sort(velodyne_pc_file.begin(), velodyne_pc_file.end());
        //* 转换到世界坐标系下合并后回到velodyne坐标系
        pcl::PointCloud<pcl::PointXYZI>::Ptr points(
            new pcl::PointCloud<pcl::PointXYZI>);
        cout << "velodyne_pc_file:" << velodyne_pc_file.size() << endl;
        for (int frame_idx = 0;
             frame_idx < std::min((int)velodyne_pc_file.size(), 1000); frame_idx++) {
            std::fstream cur_pc(velodyne_pc_file[frame_idx],
                                std::ios::binary | std::ios::in);
            for (int i = 0; cur_pc.good() && !cur_pc.eof(); i++) {
                Vector4f cur_xyzi;
                float input[4];
                // cout << "reading bin data" << endl;
                cur_pc.read((char*)&input[0], sizeof(float) * 4);
                cur_xyzi << input[0], input[1], input[2], 1;
                // cout << "point transormation...";
                cur_xyzi = T_velo_cam0 * T_cam0_w * T_w_cam[frame_idx] *
                           T_cam0_velo * cur_xyzi;
                cur_xyzi[3] = input[3];
                // cout << "finish" << endl;

                pcl::PointXYZI point;
                point.x = cur_xyzi(0);
                point.y = cur_xyzi(1);
                point.z = cur_xyzi(2);
                point.intensity = cur_xyzi(3);
                points->push_back(point);
            }
            cout << "sequence:" << seq << "  transormation(" << frame_idx << '/'
                 << velodyne_pc_file.size() << ") finish" << endl;
        }
        cout << "all transormation finish" << endl;
        pcl::PCDWriter writer;
        writer.write<pcl::PointXYZI>(output_path, *points, false);
    }
}
