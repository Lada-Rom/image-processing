#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> getImages(int n) {
    std::vector<cv::Mat> images;
    for (size_t i{ 1 }; i <= n; ++i)
        images.push_back(cv::imread(
            "../../../data/lab05.src" + std::to_string(i) + ".png",
            cv::IMREAD_GRAYSCALE));
    return images;
}

std::vector<std::vector<cv::Point2f>> getSourcePoints() {
    std::vector<std::vector<cv::Point2f>> points;
    points.push_back({ { 2428, 3370 }, { 250, 3465 }, { 192, 758 }, { 2357, 702 } });
    points.push_back({ { 410, 575 }, { 1903, 557 }, { 1908, 2436 }, { 410, 2426 } });
    points.push_back({ { 546, 1962 }, { 590, 479 }, { 2484, 389 }, { 2481, 2042 } });
    points.push_back({ { 674, 1932 }, { 734, 578 }, { 2457, 398 }, { 2495, 2042 } });
    points.push_back({ { 584, 1764 }, { 671, 383 }, { 2453, 380 }, { 2382, 1918 } });

    return points;
}

void drawPoints(std::vector<cv::Mat>& src_imgs, cv::Mat& dst_img,
    const std::vector<std::vector<cv::Point2f>>& src_keypoints,
    const std::vector<cv::Point2f>& dst_keypoints, bool write = false) {
    for (size_t i{}; i < src_imgs.size(); ++i)
        for (size_t j{}; j < src_keypoints[i].size(); ++j) {
//            std::cout << j << " " << src_keypoints[i].size() << " " << (j + 1) % src_keypoints[i].size() << std::endl;
            cv::line(src_imgs[i], src_keypoints[i][j], src_keypoints[i][(j + 1) % src_keypoints[i].size()], 0);
        }
    for (size_t j{}; j < dst_keypoints.size(); ++j)
        cv::line(dst_img, dst_keypoints[j], dst_keypoints[(j + 1) % dst_keypoints.size()], 0);

    if (write) {
        for (size_t i{ 1 }; i <= src_imgs.size(); ++i)
            cv::imwrite("../../../data/lab05.pnt" + std::to_string(i) + ".png", src_imgs[i - 1]);
        cv::imwrite("../../../data/lab05.pnt0.png", dst_img);
    }
}

std::vector<cv::Mat> findMultipleHomography(
    const std::vector<std::vector<cv::Point2f>>& srcs, const std::vector<cv::Point2f>& dst) {
    std::vector<cv::Mat> Hs;
    for (size_t i{}; i < srcs.size(); ++i)
        Hs.push_back(cv::findHomography(srcs[i], dst));
    return Hs;
}

void warpMultiplePerspective(
    const std::vector<cv::Mat>& srcs, std::vector<cv::Mat>& dsts,
    const std::vector<cv::Mat>& Hs, const cv::Size& size, bool write = false) {
    for (size_t i{}; i < srcs.size(); ++i)
        cv::warpPerspective(srcs[i], dsts[i], Hs[i], size);
    if (write)
        for (size_t i{1}; i <= dsts.size(); ++i)
            cv::imwrite("../../../data/lab05.dst" + std::to_string(i) + ".png", dsts[i - 1]);
}

void MIPTBinarization(
    const cv::Mat& source_image, cv::Mat& bin_image,
    size_t radius, double sigma, int d0, int thres) {
    cv::Rect rect;

    cv::Mat G = cv::Mat::zeros(source_image.size(), CV_8UC1);
    std::cout << "Matrix G" << std::endl;
    cv::GaussianBlur(
        source_image, G, cv::Size(2 * radius + 1, 2 * radius + 1), sigma);

    std::cout << "Matrix M" << std::endl;
    cv::Mat M = cv::Mat::zeros(source_image.size(), CV_8UC1);
    double source_pixel, gauss_pixel;
    for (size_t i{}; i < M.cols; ++i) {
        for (size_t j{}; j < M.rows; ++j) {
            source_pixel = static_cast<double>(
                source_image.at<uchar>(cv::Point(i, j)));
            gauss_pixel = static_cast<double>(
                G.at<uchar>(cv::Point(i, j)));
            M.at<uchar>(cv::Point(i, j)) = std::abs(gauss_pixel - source_pixel);
        }
    }

    std::cout << "Matrix D" << std::endl;
    cv::Mat D = cv::Mat::zeros(source_image.size(), CV_8UC1);
    cv::GaussianBlur(M, D, cv::Size(2 * radius + 1, 2 * radius + 1), sigma);

    std::cout << "Matrix B" << std::endl;
    cv::Mat B = cv::Mat::zeros(source_image.size(), CV_8UC1);
    double pixelI, pixelG, pixelD;
    double condition;
    for (size_t i{}; i < B.cols; ++i) {
        for (size_t j{}; j < B.rows; ++j) {
            pixelG = static_cast<double>(
                G.at<uchar>(cv::Point(i, j)));
            pixelI = static_cast<double>(
                source_image.at<uchar>(cv::Point(i, j)));
            pixelD = static_cast<double>(
                D.at<uchar>(cv::Point(i, j)));
            condition = (pixelG - pixelI) / (pixelD + d0);
            B.at<uchar>(cv::Point(i, j)) = (255 * condition < thres) ? 255 : 0;
        }
    }

    bin_image = B;
}

void writeBinarized(const std::vector<cv::Mat>& srcs, const cv::Mat& src,
    std::vector<cv::Mat>& dsts, cv::Mat& dst, bool write = false) {
    std::cout << "image 1" << std::endl;
    MIPTBinarization(srcs[0], dsts[0], 20, 15, 20, 200);
    std::cout << "image 2" << std::endl;
    MIPTBinarization(srcs[1], dsts[1], 20, 40, 20, 180);
    std::cout << "image 3" << std::endl;
    MIPTBinarization(srcs[2], dsts[2], 20, 15, 20, 150);
    std::cout << "image 4" << std::endl;
    MIPTBinarization(srcs[3], dsts[3], 20, 50, 20, 150);
    std::cout << "image 5" << std::endl;
    MIPTBinarization(srcs[4], dsts[4], 20, 60, 20, 150);
    cv::threshold(src, dst, 127, 255, cv::THRESH_BINARY);

    if (write) {
        for (size_t i{}; i < srcs.size(); ++i)
            cv::imwrite("../../../data/lab05.bin" + std::to_string(i + 1) + ".png", dsts[i]);
        cv::imwrite("../../../data/lab05.bin0.png", dst);
    }
}

cv::Mat deviation(const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat deviation(lhs.size(), CV_8UC3);
    double MSE{};
    double PSNR{};

    double lhs_pixel, rhs_pixel, error;
    for (size_t i{}; i < lhs.cols; ++i) {
        for (size_t j{}; j < lhs.rows; ++j) {
            lhs_pixel = static_cast<double>(lhs.at<uchar>(cv::Point(i, j)));
            rhs_pixel = static_cast<double>(rhs.at<uchar>(cv::Point(i, j)));
            error = lhs_pixel - rhs_pixel;
            MSE += error * error;

            if (error < 0) {
                deviation.at<cv::Vec3b>(j, i) = cv::Vec3b{ 0, 255, 0 };
                continue;
            }
            if (error == 0) {
                deviation.at<cv::Vec3b>(j, i) = cv::Vec3b{ 0, 0, 0 };
                continue;
            }
            if (error > 0) {
                deviation.at<cv::Vec3b>(j, i) = cv::Vec3b{ 0, 0, 255 };
                continue;
            }
        }
    }

    MSE /= lhs.cols * lhs.rows;
    PSNR = 10 * std::log(255 * 255 / MSE);
    std::cout << "MSE: " << MSE << "\nPSNR: " << PSNR << std::endl;

    return deviation;
}

void writeDeviation(const std::vector<cv::Mat>& bins,
    std::vector<cv::Mat>& devs, const cv::Mat& ideal, bool write = false) {
    for (size_t i{}; i < bins.size(); ++i) {
        std::cout << "image " << i + 1 << std::endl;
        devs[i] = deviation(bins[i], ideal);
        if (write)
            cv::imwrite("../../../data/lab05.dev" + std::to_string(i + 1) + ".png", devs[i]);
    }
}

size_t connectedComponentsWithStatsColored( const cv::Mat& source) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    int num_labels = cv::connectedComponentsWithStats(
        source, labels, stats, centroids);

    return num_labels;
}

void multiConnectedComponents(const std::vector<cv::Mat>& srcs) {
    std::vector<cv::Mat> source_inverted{srcs.size()};
    for (size_t i{}; i < srcs.size(); ++i)
        cv::threshold(srcs[i], source_inverted[i], 127, 255, cv::THRESH_BINARY_INV);
    for (size_t i{}; i < srcs.size(); ++i) {
        std::cout << "image" << i + 1 << std::endl;
        std::cout << "components: " << connectedComponentsWithStatsColored(srcs[i]) << std::endl;
    }
}

int main() {
    //images[0] is ideal image
    std::cout << "Reading..." << std::endl;
    size_t num_images = 5;
    std::vector<cv::Mat> source_images = getImages(num_images);
    cv::Mat target_image = cv::imread("../../../data/lab05.src0.png", cv::IMREAD_GRAYSCALE);

    //adding keypoints for homography
    std::cout << "\nAdding keypoints..." << std::endl;
    std::vector<std::vector<cv::Point2f>> source_points = getSourcePoints();
    std::vector<cv::Point2f> target_points = { {93, 106}, {2264, 106}, {2264, 2785}, {93, 2785} };

    //showing keypoints
    std::cout << "\nShowing keypoints..." << std::endl;
    std::vector<cv::Mat> src_visual_keypoints(source_images.size());
    for (size_t i{}; i < source_images.size(); ++i)
        source_images[i].copyTo(src_visual_keypoints[i]);
    cv::Mat dst_visual_keypoints;
    target_image.copyTo(dst_visual_keypoints);
    drawPoints(src_visual_keypoints, dst_visual_keypoints, source_points, target_points, true);

    //finding homography
    std::cout << "\nFinding homography..." << std::endl;
    std::vector<cv::Mat> Hs = findMultipleHomography(source_points, target_points);

    //warping
    std::cout << "\nWarping..." << std::endl;
    std::vector<cv::Mat> warped_images{num_images};
    warpMultiplePerspective(source_images, warped_images, Hs, target_image.size(), true);

    //binarization
    std::cout << "\nBinarizing..." << std::endl;
    std::vector<cv::Mat> source_bin_images{num_images};
    cv::Mat target_bin_image;
    writeBinarized(warped_images, target_image, source_bin_images, target_bin_image, true);

    //deviation
    std::cout << "\nCalculating deviation..." << std::endl;
    std::vector<cv::Mat> source_deviation_images{num_images};
    writeDeviation(source_bin_images, source_deviation_images, target_bin_image, true);

    //components
    std::cout << "\nCalculating components..." << std::endl;
    multiConnectedComponents(source_bin_images);
    std::cout << "ideal: " << connectedComponentsWithStatsColored(target_bin_image) << std::endl;

    cv::waitKey(0);
}
