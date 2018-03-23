#include <keyboard_lib.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Eigen>

#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#define W 640
#define H 480

#define PLANE_OFFSET 0.004

template <typename type>
type sqr(type arg){
    return arg * arg;
}

namespace pcl{
    template <typename type>
    boost::shared_ptr<type> make_shared(type& arg){
        return boost::make_shared<type>(arg);
    }
}

namespace pcl{
    template<typename type>
    void computePlane(pcl::PointCloud<type>& points, pcl::ModelCoefficients::Ptr planeCoefficients){
        if(!planeCoefficients) return;

        planeCoefficients->values.assign(4, 0);
        planeCoefficients->values[2] = 1;

        if(points.size() < 3) return;

        Eigen::Vector4f model_coefficients;
        EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
        Eigen::Vector4f xyz_centroid;

        computeMeanAndCovarianceMatrix(points, covariance_matrix, xyz_centroid);

        EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_value;
        EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;
        eigen33(covariance_matrix, eigen_value, eigen_vector);

        model_coefficients[0] = eigen_vector[0];
        model_coefficients[1] = eigen_vector[1];
        model_coefficients[2] = eigen_vector[2];
        model_coefficients[3] = 0;
        model_coefficients[3] = -1 * model_coefficients.dot(xyz_centroid);

        for(int i = 0; i < 4; i++)
            planeCoefficients->values[i] = model_coefficients[i];
    }
}

unsigned int connectedComponents(cv::Mat& inputMat, cv::Mat* outputMat = NULL, std::vector<std::vector<cv::Point>>* componentsPoints = NULL){
    unsigned int componentsNumber = 0;
    static cv::Mat compIndexMat;
    static cv::Mat inputMatMask;
    inputMatMask = cv::Mat(inputMat.size(), CV_8UC1, cv::Scalar(0));
    inputMatMask.setTo(cv::Scalar(1), inputMat);
    if(!outputMat)
        outputMat = &compIndexMat;
    *outputMat = cv::Mat(inputMatMask.size(), CV_32SC1, cv::Scalar(-1));

    static std::vector<cv::Point> queue;
    if(componentsPoints)
        componentsPoints->clear();
    for(int ix = 0; ix < inputMatMask.cols; ix++)
        for(int iy = 0; iy < inputMatMask.rows; iy++){
            if(inputMatMask.at<uint8_t>(cv::Point(ix, iy)) == 0) continue;
            int& pixelIndice = outputMat->at<int>(cv::Point(ix, iy));
            if(pixelIndice != -1) continue;
            queue.clear();
            queue.push_back(cv::Point(ix, iy));
            if(componentsPoints)
                componentsPoints->push_back(std::vector<cv::Point>(0));
            while(!queue.empty()){
                cv::Point queuePixel = queue.front();
                queue.erase(queue.begin());
                for(int dx = -1; dx <= 1; dx++)
                    for(int dy = -1; dy <= 1; dy++){
                        if(dx == 0 && dy == 0) continue;
                        cv::Point currentPixel = queuePixel + cv::Point(dx, dy);
                        if(!(currentPixel.x >= 0 && currentPixel.x < inputMatMask.cols && currentPixel.y >= 0 && currentPixel.y < inputMatMask.rows)) continue;
                        if(inputMatMask.at<uint8_t>(currentPixel) == 0) continue;
                        int& currentPixelIndice = outputMat->at<int>(cv::Point(currentPixel.x, currentPixel.y));
                        if(currentPixelIndice != -1) continue;
                        currentPixelIndice = componentsNumber;
                        queue.push_back(currentPixel);
                        if(componentsPoints)
                            componentsPoints->back().push_back(currentPixel);
                    }
            }
            componentsNumber++;
        }
    return componentsNumber;
}

void getRealsenseXYZRGBCloud(rs2::pipeline& sensorStream, rs2::frameset& frames, pcl::PointCloud<pcl::PointXYZ>& pointCloudXYZ, pcl::PointCloud<pcl::PointXYZRGB>& pointCloudXYZRGB){
    pointCloudXYZRGB.clear();

    rs2::depth_frame depth_frame = frames.first_or_default(RS2_STREAM_DEPTH);
    rs2::video_frame color_frame = frames.first_or_default(RS2_STREAM_COLOR);

    if(!(depth_frame && color_frame)) return;

    static rs2_extrinsics depthToColor = sensorStream.get_active_profile().get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(sensorStream.get_active_profile().get_stream(RS2_STREAM_COLOR));
    static rs2_intrinsics colorIntrin = sensorStream.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    float depthPoint[3];
    float colorPoint[3];
    float colorPixel[2];

    for(pcl::PointXYZ pointXYZ : pointCloudXYZ){
        depthPoint[0] = pointXYZ.x;
        depthPoint[1] = pointXYZ.y;
        depthPoint[2] = pointXYZ.z;
        rs2_transform_point_to_point(colorPoint, &depthToColor, depthPoint);
        rs2_project_point_to_pixel(colorPixel, &colorIntrin, colorPoint);

        int colorPixelX = round(colorPixel[0]);
        int colorPixelY = round(colorPixel[1]);
        if(!(colorPixelX >= 0 && colorPixelX < color_frame.get_width() && colorPixelY >= 0 && colorPixelY < color_frame.get_height())) continue;

        uint8_t* color = (uint8_t*)color_frame.get_data() + colorPixelY * color_frame.get_stride_in_bytes() + colorPixelX * color_frame.get_bytes_per_pixel();
        pcl::PointXYZRGB pointXYZRGB;
        pcl::copyPoint(pointXYZ, pointXYZRGB);
        pointXYZRGB.r = color[2];
        pointXYZRGB.g = color[1];
        pointXYZRGB.b = color[0];
        pointCloudXYZRGB.push_back(pointXYZRGB);
    }
}

int main (int argc, char** argv) {

    Keyboard keyboard;

    rs2::config cfg;
    cfg.disable_all_streams();
    cfg.enable_stream(RS2_STREAM_COLOR, 0, W, H, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, 60);
    cfg.enable_stream(RS2_STREAM_INFRARED, W, H, RS2_FORMAT_Y16, 60);

    rs2::pipeline pipe;
    rs2::pipeline_profile selection = pipe.start(cfg);
    rs2::depth_sensor depthSensor = pipe.get_active_profile().get_device().first<rs2::depth_sensor>();
    rs2_intrinsics colorIntrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2_intrinsics depthIntrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    rs2_extrinsics depthToColorExtrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(pipe.get_active_profile().get_stream(RS2_STREAM_COLOR));

    depthSensor.set_option(RS2_OPTION_CONFIDENCE_THRESHOLD, depthSensor.get_option_range(RS2_OPTION_CONFIDENCE_THRESHOLD).max);
    depthSensor.set_option(RS2_OPTION_ACCURACY, depthSensor.get_option_range(RS2_OPTION_ACCURACY).max);


    pcl::visualization::PCLVisualizer cloudViewer("Cloud viewer");
    cloudViewer.setBackgroundColor(0, 0, 0);
    cloudViewer.addCoordinateSystem(0.01);
    cloudViewer.initCameraParameters();
    cloudViewer.setCameraPosition(0, 0, 0, 0, -1, -1);

    bool mode = 0;

    while (!cloudViewer.wasStopped() && !keyboard.keyPush(KEY_ESC))
    {
        boost::posix_time::ptime t0 = boost::posix_time::microsec_clock::local_time();

        static bool updateEnable = true;
        if(keyboard.keyPush(KEY_SPACE))
            updateEnable = !updateEnable;

        if(updateEnable) {
            cloudViewer.removeAllPointClouds();
            cloudViewer.removeAllShapes();

            rs2::frameset frames = pipe.wait_for_frames();
            rs2::depth_frame depth_frame = frames.first_or_default(RS2_STREAM_DEPTH);
            rs2::video_frame color_frame = frames.first_or_default(RS2_STREAM_COLOR);
            rs2::video_frame infrared_frame = frames.first_or_default(RS2_STREAM_INFRARED);

            if (depth_frame && color_frame) {
                static pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorImagePointCloud(new pcl::PointCloud<pcl::PointXYZRGB>(W, H));

                static pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointCloudXYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr filteredPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredPointCloudXYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr planeConvexHullPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr workFieldPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr planePointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr objectsFieldPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                static pcl::PointCloud<pcl::PointXYZ>::Ptr voxelGridProjectedPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);

                static std::vector<pcl::PointCloud<pcl::PointXYZ>> contoursPointCloudsXYZ;
                static std::vector<pcl::PointCloud<pcl::PointXYZ>> objectsPointCloudsXYZ;
                static std::vector<pcl::PointXYZ> objectsCenter;

                static pcl::VoxelGrid<pcl::PointXYZ> voxelGridFilter;
                static pcl::SACSegmentation<pcl::PointXYZ> sacSegmentation;
                static pcl::ExtractIndices<pcl::PointXYZ> extractIndices;
                static pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClusterExtraction;
                static pcl::ProjectInliers<pcl::PointXYZ> projectInliers;
                static pcl::ExtractPolygonalPrismData<pcl::PointXYZ> extractPolygonalPrismData;
                static pcl::ConvexHull<pcl::PointXYZ> convexHull;
                static pcl::CropHull<pcl::PointXYZ> cropHull;
                static pcl::MomentOfInertiaEstimation<pcl::PointXYZ> featureExtractor;

                static pcl::ModelCoefficients::Ptr workPlaneCoefficientsPCL(new pcl::ModelCoefficients);
                static pcl::PointIndices::Ptr pointIndices(new pcl::PointIndices);
                static pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);

                static std::vector<pcl::PointIndices> clusterPointsIndices;

                sourcePointCloudXYZ->clear();
                for (pcl::PointXYZRGB& point : *colorImagePointCloud) {
                    point.x = 0;
                    point.y = 0;
                    point.z = 0;
                }

                for (int ix = 0; ix < depth_frame.get_width(); ix++)
                    for (int iy = 0; iy < depth_frame.get_height(); iy++) {
                        float dist = depth_frame.get_distance(ix, iy);
                        static float depthPoint[3];
                        static float colorPoint[3];
                        static float depthPixel[2];
                        static float colorPixel[2];
                        depthPixel[0] = ix;
                        depthPixel[1] = iy;

                        if (dist > 0) {
                            rs2_deproject_pixel_to_point(depthPoint, &depthIntrinsics, depthPixel, dist);
                            rs2_transform_point_to_point(colorPoint, &depthToColorExtrinsics, depthPoint);
                            rs2_project_point_to_pixel(colorPixel, &colorIntrinsics, colorPoint);
                            sourcePointCloudXYZ->push_back(pcl::PointXYZ(depthPoint[0], depthPoint[1], depthPoint[2]));

                            int colorPixelX = round(colorPixel[0]);
                            int colorPixelY = round(colorPixel[1]);
                            if (!(colorPixelX >= 0 && colorPixelX < color_frame.get_width() && colorPixelY >= 0 &&
                                  colorPixelY < color_frame.get_height()))
                                continue;

                            uint8_t *color =
                                    (uint8_t *) color_frame.get_data() +
                                    colorPixelY * color_frame.get_stride_in_bytes() +
                                    colorPixelX * color_frame.get_bytes_per_pixel();

                            colorImagePointCloud->at(colorPixelX, colorPixelY).x = depthPoint[0];
                            colorImagePointCloud->at(colorPixelX, colorPixelY).y = depthPoint[1];
                            colorImagePointCloud->at(colorPixelX, colorPixelY).z = depthPoint[2];
                            colorImagePointCloud->at(colorPixelX, colorPixelY).r = color[2];
                            colorImagePointCloud->at(colorPixelX, colorPixelY).g = color[1];
                            colorImagePointCloud->at(colorPixelX, colorPixelY).b = color[0];
                        }
                    }

                for (int ix = 0; ix < colorImagePointCloud->width; ix++)
                    for (int iy = 0; iy < colorImagePointCloud->height; iy++) {
                        pcl::PointXYZRGB &point = colorImagePointCloud->at(ix, iy);
                        if (point.z == 0) {
                            uint8_t *color =
                                    (uint8_t *) color_frame.get_data() + iy * color_frame.get_stride_in_bytes() +
                                    ix * color_frame.get_bytes_per_pixel();
                            point.r = color[2];
                            point.g = color[1];
                            point.b = color[0];
                        }
                    }

                getRealsenseXYZRGBCloud(pipe, frames, *sourcePointCloudXYZ, *sourcePointCloudXYZRGB);

                if (keyboard.keyPush(KEY_ENTER) || 1) {

                    voxelGridFilter.setInputCloud(sourcePointCloudXYZ);
                    float voxelGridSize = 0.01f;
                    voxelGridFilter.setLeafSize(voxelGridSize, voxelGridSize, voxelGridSize);
                    voxelGridFilter.filter(*filteredPointCloudXYZ);

                    sacSegmentation.setOptimizeCoefficients(true);
                    sacSegmentation.setModelType(pcl::SACMODEL_PLANE);
                    sacSegmentation.setMethodType(pcl::SAC_RANSAC);
                    sacSegmentation.setMaxIterations(500);
                    sacSegmentation.setDistanceThreshold(0.005);
                    sacSegmentation.setInputCloud(filteredPointCloudXYZ);
                    sacSegmentation.segment(*pointIndices, *workPlaneCoefficientsPCL);

                    if (pointIndices->indices.empty())continue;

                    std::pair<Eigen::Vector3d, double> planeCoefficients;
                    planeCoefficients.first
                            << workPlaneCoefficientsPCL->values[0], workPlaneCoefficientsPCL->values[1], workPlaneCoefficientsPCL->values[2];
                    planeCoefficients.second = workPlaneCoefficientsPCL->values[3];
                    planeCoefficients.second /= planeCoefficients.first.norm();
                    planeCoefficients.first.normalize();
                    if (planeCoefficients.first.z() > 0) {
                        planeCoefficients.first = -planeCoefficients.first;
                        planeCoefficients.second = -planeCoefficients.second;
                    }
                    if(planeCoefficients.first.z() == 0) continue;

                    extractIndices.setInputCloud(filteredPointCloudXYZ);
                    extractIndices.setIndices(pointIndices);
                    extractIndices.setNegative(false);
                    extractIndices.filter(*filteredPointCloudXYZ);

                    projectInliers.setModelType(pcl::SACMODEL_PLANE);
                    projectInliers.setInputCloud(filteredPointCloudXYZ);
                    projectInliers.setModelCoefficients(workPlaneCoefficientsPCL);
                    projectInliers.filter(*filteredPointCloudXYZ);

                    getRealsenseXYZRGBCloud(pipe, frames, *filteredPointCloudXYZ, *filteredPointCloudXYZRGB);
                    pcl::copyPointCloud(*filteredPointCloudXYZRGB, *filteredPointCloudXYZ);

                    kdTree->setInputCloud(filteredPointCloudXYZ);
                    clusterPointsIndices.clear();
                    euclideanClusterExtraction.setClusterTolerance(0.02);
                    euclideanClusterExtraction.setMinClusterSize(filteredPointCloudXYZ->size() * 0.3);
                    euclideanClusterExtraction.setMaxClusterSize(filteredPointCloudXYZ->size());
                    euclideanClusterExtraction.setSearchMethod(kdTree);
                    euclideanClusterExtraction.setInputCloud(filteredPointCloudXYZ);
                    euclideanClusterExtraction.extract(clusterPointsIndices);

                    if (clusterPointsIndices.empty())
                        continue;

                    planePointCloudXYZ->clear();
                    for (pcl::PointIndices &currentClusterPointsIndices : clusterPointsIndices)
                        for (int indice : currentClusterPointsIndices.indices)
                            planePointCloudXYZ->push_back(filteredPointCloudXYZ->at(indice));

                    convexHull.setInputCloud(planePointCloudXYZ);
                    convexHull.setDimension(2);
                    convexHull.reconstruct(*planeConvexHullPointCloud);

                    if(planeConvexHullPointCloud->empty())
                        continue;

                    static std::vector<cv::Point> convexHullPixels;
                    convexHullPixels.clear();

                    {
                        static pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTreeFLANN;
                        kdTreeFLANN.setInputCloud(colorImagePointCloud);
                        for (pcl::PointXYZ point : *planeConvexHullPointCloud) {
                            static std::vector<int> pointsIndices;
                            static std::vector<float> pointsDistances;
                            pcl::PointXYZRGB searchPoint;
                            pcl::copyPoint(point, searchPoint);
                            if (kdTreeFLANN.nearestKSearch(searchPoint, 1, pointsIndices, pointsDistances) == 0)
                                continue;
                            convexHullPixels.push_back(cv::Point(pointsIndices.back() % colorImagePointCloud->width,
                                                                 pointsIndices.back() / colorImagePointCloud->width));
                        }
                    }

                    if (!cv::isContourConvex(convexHullPixels))
                        cv::convexHull(convexHullPixels, convexHullPixels);

                    extractPolygonalPrismData.setInputCloud(sourcePointCloudXYZ);
                    extractPolygonalPrismData.setInputPlanarHull(planeConvexHullPointCloud);
                    extractPolygonalPrismData.setHeightLimits(-PLANE_OFFSET, INFINITY);
                    extractPolygonalPrismData.segment(*pointIndices);

                    extractIndices.setInputCloud(sourcePointCloudXYZ);
                    extractIndices.setIndices(pointIndices);
                    extractIndices.filter(*workFieldPointCloudXYZ);

                    extractPolygonalPrismData.setInputCloud(workFieldPointCloudXYZ);
                    extractPolygonalPrismData.setInputPlanarHull(planeConvexHullPointCloud);
                    extractPolygonalPrismData.setHeightLimits(-PLANE_OFFSET, PLANE_OFFSET);
                    extractPolygonalPrismData.segment(*pointIndices);

                    extractIndices.setInputCloud(workFieldPointCloudXYZ);
                    extractIndices.setIndices(pointIndices);
                    extractIndices.filter(*planePointCloudXYZ);

                    extractPolygonalPrismData.setInputCloud(workFieldPointCloudXYZ);
                    extractPolygonalPrismData.setInputPlanarHull(planeConvexHullPointCloud);
                    extractPolygonalPrismData.setHeightLimits(PLANE_OFFSET, INFINITY);
                    extractPolygonalPrismData.segment(*pointIndices);

                    extractIndices.setInputCloud(workFieldPointCloudXYZ);
                    extractIndices.setIndices(pointIndices);
                    extractIndices.filter(*objectsFieldPointCloudXYZ);


                    cv::Mat colorImage(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3,
                                       (void *) color_frame.get_data(), cv::Mat::AUTO_STEP);
                    static cv::Mat workFieldMask(cv::Size(colorImage.cols, colorImage.rows), CV_8UC1);
                    static cv::Mat workImage(cv::Size(colorImage.cols, colorImage.rows), CV_8UC3);
                    workFieldMask = cv::Scalar(0);
                    cv::fillConvexPoly(workFieldMask, convexHullPixels, cv::Scalar(255));
                    workImage = CV_RGB(0, 255, 0);
                    colorImage.copyTo(workImage, workFieldMask);

                    static cv::Mat cannyFieldMask = cv::Mat(workFieldMask.size(), CV_8UC1);
                    static cv::Mat workFieldMaskChange = cv::Mat(workFieldMask.size(), CV_8UC1);;
                    cannyFieldMask = cv::Scalar(0);
                    cv::dilate(workFieldMask, workFieldMaskChange,
                               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)),
                               cv::Point(-1, -1), 1);
                    cannyFieldMask.setTo(cv::Scalar(255), workFieldMaskChange - workFieldMask);
                    cv::erode(workFieldMask, workFieldMaskChange,
                              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)),
                              cv::Point(-1, -1), 1);
                    cannyFieldMask.setTo(cv::Scalar(255), workFieldMask - workFieldMaskChange);
                    cv::dilate(cannyFieldMask, cannyFieldMask,
                               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(19, 19), cv::Point(-1, -1)),
                               cv::Point(-1, -1), 1);
                    cannyFieldMask |= ~workFieldMask;
                    cannyFieldMask = ~cannyFieldMask;

                    static cv::Mat cannyWorkImage;
                    static cv::Mat cannyImage;
                    cv::cvtColor(workImage, cannyImage, CV_RGB2GRAY);
                    int threshold = 50;
                    cv::Canny(cannyImage, cannyImage, threshold, threshold * 5, 3);
                    cannyWorkImage = cannyImage & cannyFieldMask;

                    for (int i = 3; i <= 11; i += 2) {
                        cv::dilate(cannyWorkImage, cannyWorkImage,
                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                   cv::Point(-1, -1), 1);
                        cv::erode(cannyWorkImage, cannyWorkImage,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                    }
                    //cv::imshow("Canny", cannyWorkImage);

                    static std::vector<std::vector<cv::Point>> clustersPixels;
                    connectedComponents(cannyWorkImage, NULL, &clustersPixels);

                    static std::vector<std::vector<cv::Point>> clustersConvexHullPixels;
                    clustersConvexHullPixels.clear();

                    for (std::vector<cv::Point> &clusterPixels : clustersPixels) {
                        if (clusterPixels.size() < cannyWorkImage.cols * cannyWorkImage.rows * 0.0005) continue;
                        clustersConvexHullPixels.push_back(std::vector<cv::Point>(0));
                        cv::convexHull(clusterPixels, clustersConvexHullPixels.back());
                    }

                    static cv::Mat workPlaneMaskFromColor;
                    workPlaneMaskFromColor = workFieldMask.clone();
                    for (std::vector<cv::Point> &clusterHull : clustersConvexHullPixels)
                        cv::fillConvexPoly(workPlaneMaskFromColor, clusterHull, cv::Scalar(0));

                    //static cv::Mat segmetatedObjects;
                    //cv::cvtColor(cannyWorkImage, segmetatedObjects, CV_GRAY2BGR);
                    //for(std::vector<cv::Point>& clusterHull : clustersConvexHullPixels)
                    //    for(cv::Point& point : clusterHull)
                    //        cv::circle(segmetatedObjects, point, 3, CV_RGB(255, 0, 0), -1);
                    //cv::imshow("Hull", segmetatedObjects);

                    static cv::Mat workPlaneMaskFromPoints(workFieldMask.size(), CV_8UC1);
                    workPlaneMaskFromPoints = cv::Scalar(0);
                    for (int ix = 0; ix < workPlaneMaskFromPoints.cols; ix++)
                        for (int iy = 0; iy < workPlaneMaskFromPoints.rows; iy++)
                            if (workFieldMask.at<uint8_t>(cv::Point(ix, iy)) != 0 &&
                                colorImagePointCloud->at(ix, iy).z > 0)
                                if (fabs(planeCoefficients.first.dot(Eigen::Vector3d(colorImagePointCloud->at(ix, iy).x,
                                                                                     colorImagePointCloud->at(ix, iy).y,
                                                                                     colorImagePointCloud->at(ix,
                                                                                                             iy).z)) +
                                         planeCoefficients.second) <= PLANE_OFFSET)
                                    workPlaneMaskFromPoints.at<uint8_t>(cv::Point(ix, iy)) = 255;

                    for (int i = 3; i <= 5; i += 2) {
                        cv::dilate(workPlaneMaskFromPoints, workPlaneMaskFromPoints,
                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                   cv::Point(-1, -1), 1);
                        cv::erode(workPlaneMaskFromPoints, workPlaneMaskFromPoints,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                    }

                    static cv::Mat workPlaneMaskTotal;
                    workPlaneMaskTotal = workPlaneMaskFromColor & workPlaneMaskFromPoints;

                    for (int i = 3; i <= 15; i += 2) {
                        cv::dilate(workPlaneMaskTotal, workPlaneMaskTotal,
                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                   cv::Point(-1, -1), 1);
                        cv::erode(workPlaneMaskTotal, workPlaneMaskTotal,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                    }

                    //workImage = cv::Scalar(0, 255, 0);
                    //colorImage.copyTo(workImage, workFieldMask);
                    //workImage.setTo(cv::Scalar(255, 0, 0), workPlaneMaskTotal & workFieldMask);
                    //cv::imshow("Segmentated objects", workImage);

                    static cv::Mat objectsMask;
                    objectsMask = workPlaneMaskTotal | (~workFieldMask);
                    for (int i = 3; i <= 15; i += 2) {
                        cv::dilate(objectsMask, objectsMask,
                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                   cv::Point(-1, -1), 1);
                        cv::erode(objectsMask, objectsMask,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                    }
                    objectsMask = (~objectsMask) & workFieldMask;
                    //cv::imshow("Objects mask", objectsMask);

                    static cv::Mat componentsIndicies;
                    static std::vector<uint8_t> badComponentsIndicies;
                    int componentsNumber;
                    do {
                        componentsNumber = connectedComponents(objectsMask, &componentsIndicies);
                        if (componentsNumber >= 255) break;
                        componentsIndicies += cv::Scalar(1);
                        componentsIndicies.convertTo(componentsIndicies, CV_8UC1);

                        //                static cv::Mat badComponents;
                        //                cv::erode(workFieldMask, workFieldMaskChange, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)), cv::Point(-1, -1), 1);
                        //                workFieldMaskChange = (workFieldMask - workFieldMaskChange);
                        //                cv::rectangle(workFieldMaskChange, cv::Point(0, 0), cv::Point(workFieldMask.cols - 1, workFieldMask.rows - 1), cv::Scalar(255), 1);
                        //                badComponents = componentsIndicies & workFieldMaskChange;
                        //                badComponentsIndicies.clear();
                        //                for(int ix = 0; ix < badComponents.cols; ix++)
                        //                    for(int iy = 0; iy < badComponents.rows; iy++) {
                        //                        uint8_t indice = badComponents.at<uint8_t>(cv::Point(ix, iy));
                        //                        if (indice != 0 &&
                        //                            std::find(badComponentsIndicies.begin(), badComponentsIndicies.end(), indice) ==
                        //                            badComponentsIndicies.end())
                        //                            badComponentsIndicies.push_back(indice);
                        //                    }
                        //                for(uint8_t indice : badComponentsIndicies)
                        //                    objectsMask.setTo(cv::Scalar(0), componentsIndicies == indice);
                    } while (!badComponentsIndicies.empty());

                    if (componentsNumber >= 254) continue;
                    //cv::imshow("Components", componentsIndicies * (255.0 / componentsNumber));
                    componentsIndicies.setTo(cv::Scalar(componentsNumber + 1), ~workFieldMask);

                    static cv::Mat pixelIniciesOld;
                    do {
                        pixelIniciesOld = componentsIndicies.clone();
                        for (int componentNumber = 1; componentNumber <= componentsNumber + 1; componentNumber++) {
                            static cv::Mat currentComponent;
                            static cv::Mat currentComponentChanges;
                            static cv::Mat currentComponentSpreading;
                            currentComponent = (componentsIndicies == componentNumber);
                            cv::threshold(currentComponent, currentComponent, 0, 255, cv::THRESH_BINARY);
                            cv::dilate(currentComponent, currentComponentChanges,
                                       cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(-1, -1)),
                                       cv::Point(-1, -1), 1);
                            currentComponentChanges -= currentComponent;
                            currentComponentChanges &= workFieldMask;
                            cv::threshold(componentsIndicies, currentComponentSpreading, 0, 255, cv::THRESH_BINARY_INV);
                            currentComponentSpreading &= currentComponentChanges;
                            componentsIndicies.setTo(cv::Scalar(componentNumber), currentComponentSpreading);
                        }
                    } while (cv::countNonZero(componentsIndicies != pixelIniciesOld) != 0);
                    componentsIndicies.setTo(cv::Scalar(0), componentsIndicies == (componentsNumber + 1));

                    for (int i = 3; i <= 7; i += 2) {
                        cv::erode(componentsIndicies, componentsIndicies,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                        cv::dilate(componentsIndicies, componentsIndicies,
                                   cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(-1, -1)),
                                   cv::Point(-1, -1), 1);
                    }
                    cv::rectangle(componentsIndicies, cv::Point(0, 0),
                                  cv::Point(componentsIndicies.cols - 1, componentsIndicies.rows - 1), cv::Scalar(0),
                                  1);
                    //cv::imshow("Components", componentsIndicies * (255.0 / componentsNumber));

                    static cv::Mat componentsBoundaries = cv::Mat(componentsIndicies.size(), CV_8UC1);
                    componentsBoundaries = cv::Scalar(0);
                    for (int componentNumber = 1; componentNumber <= componentsNumber; componentNumber++) {
                        static cv::Mat currentComponent;
                        static cv::Mat currentComponentChanges;
                        currentComponent = (componentsIndicies == componentNumber);
                        cv::threshold(currentComponent, currentComponent, 0, 255, cv::THRESH_BINARY);
                        cv::erode(currentComponent, currentComponentChanges,
                                  cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1)),
                                  cv::Point(-1, -1), 1);
                        cv::rectangle(currentComponentChanges, cv::Point(0, 0),
                                      cv::Point(currentComponentChanges.cols - 1, currentComponentChanges.rows - 1),
                                      cv::Scalar(0), 1);
                        componentsBoundaries.setTo(cv::Scalar(componentNumber),
                                                   currentComponent != currentComponentChanges);
                    }
                    //cv::imshow("Boundaries", componentsBoundaries * (255.0 / componentsNumber));

                    static std::vector<std::vector<cv::Point>> componentsContours;
                    componentsContours.assign(componentsNumber, std::vector<cv::Point>(0));
                    static cv::Mat componentsContoursImage(componentsBoundaries.size(), CV_8UC1);
                    componentsContoursImage = cv::Scalar(0);
                    for (uint8_t componentNumber = 1; componentNumber <= componentsNumber; componentNumber++) {
                        static std::vector<std::vector<cv::Point>> contours;
                        static std::vector<cv::Vec4i> hierarchy;
                        static cv::Mat currentBoundary;
                        contours.clear();
                        hierarchy.clear();
                        currentBoundary = (componentsBoundaries == componentNumber);
                        cv::findContours(currentBoundary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

                        if (contours.empty()) continue;
                        int maxContourNumber = 0;
                        for (int i = 0; i < contours.size(); i++)
                            if (contours[i].size() > contours[maxContourNumber].size())
                                maxContourNumber = i;

                        componentsContours[componentNumber - 1].assign(contours[maxContourNumber].begin(),
                                                                       contours[maxContourNumber].end());
                        for (cv::Point point : componentsContours[componentNumber - 1])
                            componentsContoursImage.at<uint8_t>(point) = componentNumber;
                    }
                    cv::imshow("Contours", componentsContoursImage * (255.0 / componentsNumber));

                    static std::vector<std::vector<pcl::PointXYZ>> objectsContours3D;
                    objectsContours3D.assign(componentsContours.size(), std::vector<pcl::PointXYZ>(0));

                    for (int i = 0; i < componentsContours.size(); i++)
                        for (cv::Point componentContourPoint : componentsContours[i]) {
                            pcl::PointXYZRGB &contourPoint = colorImagePointCloud->at(componentContourPoint.x,
                                                                                     componentContourPoint.y);
                            if (contourPoint.z > 0 && fabs(planeCoefficients.first.dot(
                                    Eigen::Vector3d(contourPoint.x, contourPoint.y, contourPoint.z)) +
                                                           planeCoefficients.second) <= PLANE_OFFSET)
                                objectsContours3D[i].push_back(
                                        pcl::PointXYZ(contourPoint.x, contourPoint.y, contourPoint.z));
                        }

                    contoursPointCloudsXYZ.assign(objectsContours3D.size(), pcl::PointCloud<pcl::PointXYZ>(0, 1));
                    for (int i = 0; i < objectsContours3D.size(); i++) {
                        contoursPointCloudsXYZ[i].points.assign(objectsContours3D[i].begin(),
                                                                objectsContours3D[i].end());

                        static pcl::PointCloud<pcl::PointXYZ> projectedContourPointCloud;
                        projectInliers.setInputCloud(contoursPointCloudsXYZ[i].makeShared());
                        projectInliers.setModelType(pcl::SACMODEL_PLANE);
                        projectInliers.setModelCoefficients(workPlaneCoefficientsPCL);
                        projectInliers.filter(contoursPointCloudsXYZ[i]);
                    }

                    static pcl::PointCloud<pcl::PointXYZ> projectedObjectsFileldPointCloud;
                    projectInliers.setInputCloud(objectsFieldPointCloudXYZ);
                    projectInliers.setModelType(pcl::SACMODEL_PLANE);
                    projectInliers.setModelCoefficients(workPlaneCoefficientsPCL);
                    projectInliers.filter(projectedObjectsFileldPointCloud);

                    objectsPointCloudsXYZ.resize(contoursPointCloudsXYZ.size());
                    for (int contourNumber = 0; contourNumber < contoursPointCloudsXYZ.size(); contourNumber++) {
                        static std::vector<pcl::Vertices> contourVertices;
                        contourVertices.resize(1);
                        contourVertices.begin()->vertices.resize(contoursPointCloudsXYZ[contourNumber].size());
                        for (int contourPointNumber = 0;
                             contourPointNumber < contoursPointCloudsXYZ[contourNumber].size(); contourPointNumber++)
                            contourVertices.begin()->vertices[contourPointNumber] = contourPointNumber;

                        std::vector<int> inliers;
                        cropHull.setInputCloud(projectedObjectsFileldPointCloud.makeShared());
                        cropHull.setHullCloud(contoursPointCloudsXYZ[contourNumber].makeShared());
                        cropHull.setHullIndices(contourVertices);
                        cropHull.setDim(2);
                        cropHull.setCropOutside(true);
                        cropHull.filter(inliers);

                        objectsPointCloudsXYZ[contourNumber].clear();
                        for (int inlier : inliers)
                            objectsPointCloudsXYZ[contourNumber].push_back(objectsFieldPointCloudXYZ->at(inlier));
                    }

                    objectsCenter.resize(objectsPointCloudsXYZ.size());
                    for (int i = 0; i < objectsPointCloudsXYZ.size(); i++) {
                        featureExtractor.setInputCloud(objectsPointCloudsXYZ[i].makeShared());
                        featureExtractor.compute();
                        Eigen::Vector3f massCenter;
                        featureExtractor.getMassCenter(massCenter);
                        objectsCenter[i].x = massCenter.x();
                        objectsCenter[i].y = massCenter.y();
                        objectsCenter[i].z = massCenter.z();

                        //double distanceFromPlane = fabs(planeCoefficients.first.dot(massCenter.cast<double>()) + planeCoefficients.second);

                        static pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statisticalOutlierRemovalFilter;
                        statisticalOutlierRemovalFilter.setInputCloud(objectsPointCloudsXYZ[i].makeShared());
                        statisticalOutlierRemovalFilter.setMeanK(20);
                        statisticalOutlierRemovalFilter.setStddevMulThresh(0.000000001);
                        statisticalOutlierRemovalFilter.filter(objectsPointCloudsXYZ[i]);

                        clusterPointsIndices.clear();
                        euclideanClusterExtraction.setInputCloud(objectsPointCloudsXYZ[i].makeShared());
                        euclideanClusterExtraction.setClusterTolerance(0.005);
                        euclideanClusterExtraction.setMinClusterSize(1);
                        euclideanClusterExtraction.setMaxClusterSize(objectsPointCloudsXYZ[i].size());
                        euclideanClusterExtraction.extract(clusterPointsIndices);

                        int maxClusterNumber = 0;
                        for(int i = 0; i < clusterPointsIndices.size(); i++)
                            if(clusterPointsIndices[i].indices.size() > clusterPointsIndices[maxClusterNumber].indices.size())
                                maxClusterNumber = i;

                        int maxPointsNumber = clusterPointsIndices[maxClusterNumber].indices.size();
                        pointIndices->indices.clear();
                        for(pcl::PointIndices& indices : clusterPointsIndices)
                            if(indices.indices.size() > maxPointsNumber * 0.2)
                                pointIndices->indices.insert(pointIndices->indices.end(), indices.indices.begin(), indices.indices.end());

                        extractIndices.setInputCloud(objectsPointCloudsXYZ[i].makeShared());
                        extractIndices.setIndices(pointIndices);
                        extractIndices.setNegative(false);
                        extractIndices.filter(objectsPointCloudsXYZ[i]);
                    }
                }

                static std::vector<pcl::RGB> colors(6);
                colors[0].r = 255;
                colors[0].g = 0;
                colors[0].b = 0;
                colors[1].r = 0;
                colors[1].g = 255;
                colors[1].b = 0;
                colors[2].r = 0;
                colors[2].g = 0;
                colors[2].b = 255;
                colors[3].r = 255;
                colors[3].g = 255;
                colors[3].b = 0;
                colors[4].r = 255;
                colors[4].g = 0;
                colors[4].b = 255;
                colors[5].r = 0;
                colors[5].g = 255;
                colors[5].b = 255;

                {
                    typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorizerXYZ;
                    typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> colorizerXYZRGB;

                    cloudViewer.addPlane(*workPlaneCoefficientsPCL, "Work plane");
                    for (int i = 0; i < objectsPointCloudsXYZ.size(); i++) {
                        static pcl::PointCloud<pcl::PointXYZRGB>::Ptr vizualizedPointCloudXYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
                        getRealsenseXYZRGBCloud(pipe, frames, objectsPointCloudsXYZ[i], *vizualizedPointCloudXYZRGB);
                        cloudViewer.addPointCloud<pcl::PointXYZRGB>(vizualizedPointCloudXYZRGB, colorizerXYZRGB(vizualizedPointCloudXYZRGB), "object" + std::to_string(i));
                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object" + std::to_string(i));

                        pcl::RGB color = colors[i % colors.size()];
                        cloudViewer.addPointCloud<pcl::PointXYZ>(contoursPointCloudsXYZ[i].makeShared(), colorizerXYZ(contoursPointCloudsXYZ[i].makeShared(), color.r, color.g, color.b), "contour" + std::to_string(i));
                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "contour" + std::to_string(i));

                        pcl::PointCloud<pcl::PointXYZ> centerPoint(1, 1, objectsCenter[i]);
                        cloudViewer.addPointCloud<pcl::PointXYZ>(centerPoint.makeShared(), colorizerXYZ(centerPoint.makeShared(), color.r, color.g, color.b), "center" + std::to_string(i));
                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "center" + std::to_string(i));
                    }
                }

                std::cout << "Iteration time: "
                          << (boost::posix_time::microsec_clock::local_time() - t0).total_milliseconds() << std::endl;
            }
        }

        cloudViewer.spinOnce();
        cv::waitKey(1);
    }
    return 0;
}