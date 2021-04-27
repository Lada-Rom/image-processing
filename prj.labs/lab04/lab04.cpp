#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void showWindow(const cv::Mat& image, std::string name = "Image",
    size_t x = 0, size_t y = 0) {
    imshow(name, image);
    cv::moveWindow(name, x, y);
}

size_t connectedComponentsWithStatsColored(
    const cv::Mat& source, cv::Mat& dest, size_t connect) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    int num_labels = cv::connectedComponentsWithStats(
        source, labels, stats, centroids);

    int x, y, w, h, a;
    dest = cv::Mat::zeros(source.size(), CV_8UC3);
    for (size_t i{}; i < source.rows; ++i)
        for (size_t j{}; j < source.cols; ++j) {
            int label = labels.at<int>(i, j);
            cv::Vec3b& pixel = dest.at<cv::Vec3b>(i, j);
            if (label == 0) {
                pixel = cv::Vec3b{ 255, 255, 255 };
                continue;
            }
            w = stats.at<int>(cv::Point(2, label));
            h = stats.at<int>(cv::Point(3, label));
            a = stats.at<int>(cv::Point(4, label));
            if (a <= 60) {
                pixel = cv::Vec3b{ 0, 0, 255 };
                continue;
            }
            pixel = cv::Vec3b{ 0, 0, 0 };
        }

    return num_labels;
}

cv::Mat deviation(cv::Mat& lhs, cv::Mat& rhs) {
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

void MIPTBinarization(
    const cv::Mat& source_image, cv::Mat& bin_image,
    size_t radius, double sigma, int d0, int thres) {
    cv::Rect rect;

    cv::Mat G = cv::Mat::zeros(source_image.size(), CV_8UC1);
    std::cout << "Matrix G" << std::endl;
    cv::GaussianBlur(
        source_image, G, cv::Size(2 * radius + 1, 2 * radius + 1), sigma);
    //    cv::imwrite("../../../data/test1.png", G);

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
    //    cv::imwrite("../../../data/test2.png", M);

    std::cout << "Matrix D" << std::endl;
    cv::Mat D = cv::Mat::zeros(source_image.size(), CV_8UC1);
    cv::GaussianBlur(M, D, cv::Size(2 * radius + 1, 2 * radius + 1), sigma);
    //    cv::imwrite("../../../data/test3.png", D);

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

int main() {
    //source image
    cv::Mat source_image = cv::imread("../../../data/lab04.src2.jpg");

    //ideal image
    cv::Mat ideal = cv::imread("../../../data/ideal.png", cv::IMREAD_GRAYSCALE);

    //convertion to gray image
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, cv::COLOR_BGR2GRAY);

    //1.1. standart binarization
    std::cout << "1.1. Standart binarization" << std::endl;
    cv::Mat standart_bin_image;
    cv::adaptiveThreshold(
        gray_image, standart_bin_image,
        255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, 8);

    //1.2. standart filtration
    std::cout << "1.2. Standart filtration" << std::endl;
    cv::Mat standart_filtered_image;
    cv::GaussianBlur(
        standart_bin_image, standart_filtered_image,
        cv::Size(3, 3), 0, 0);
    cv::threshold(
        standart_filtered_image, standart_filtered_image,
        100, 255, cv::ThresholdTypes::THRESH_BINARY);

    //1.3. standart components
    std::cout << "1.3. Standart components" << std::endl;
    cv::Mat standart_labeled_image;
    cv::Mat standart_inverted_filtered_image;
    cv::threshold(
        standart_filtered_image, standart_inverted_filtered_image,
        100, 255, cv::ThresholdTypes::THRESH_BINARY_INV);
    size_t comp_num1 = connectedComponentsWithStatsColored(
        standart_inverted_filtered_image, standart_labeled_image, 4);
    std::cout << "Number of components: " << comp_num1 << std::endl;

    //1.4. standart deviation from the ideal
    std::cout << "1.4. Standart deviation" << std::endl;
    cv::Mat standart_deviation = deviation(standart_filtered_image, ideal);

    //2.1. gauss binarization
    std::cout << "2.1. Gauss binarization" << std::endl;
    cv::Mat gauss_bin_image;
    MIPTBinarization(gray_image, gauss_bin_image, 20, 15, 20, 200);

    //2.2. gauss filtration
    std::cout << "2.2. Gauss filtration" << std::endl;
    cv::Mat gauss_filtered_image;
    cv::GaussianBlur(
        gauss_bin_image, gauss_filtered_image, cv::Size(5, 5), 0.8);
    cv::threshold(
        gauss_filtered_image, gauss_filtered_image, 140, 255,
        cv::ThresholdTypes::THRESH_BINARY);

    //2.3. gauss components
    std::cout << "2.3. Gauss components" << std::endl;
    cv::Mat gauss_labeled_image;
    cv::Mat gauss_inverted_filtered_image;
    cv::threshold(
        gauss_filtered_image, gauss_inverted_filtered_image,
        100, 255, cv::ThresholdTypes::THRESH_BINARY_INV);
    size_t comp_num2 = connectedComponentsWithStatsColored(
        gauss_inverted_filtered_image, gauss_labeled_image, 4);
    std::cout << "Number of components: " << comp_num2 << std::endl;

    //2.4. gauss deviation from the ideal
    std::cout << "2.4. Gauss deviation" << std::endl;
    cv::Mat gauss_deviation = deviation(gauss_filtered_image, ideal);

    //visualization
    cv::imwrite("../../../data/lab04.g1.png", gray_image);

    cv::imwrite("../../../data/lab04.b1.png", standart_bin_image);
    cv::imwrite("../../../data/lab04.f1.png", standart_filtered_image);
    cv::imwrite("../../../data/lab04.v1.png", standart_labeled_image);
    cv::imwrite("../../../data/lab04.e1.png", standart_deviation);

    cv::imwrite("../../../data/lab04.b2.png", gauss_bin_image);
    cv::imwrite("../../../data/lab04.f2.png", gauss_filtered_image);
    cv::imwrite("../../../data/lab04.v2.png", gauss_labeled_image);
    cv::imwrite("../../../data/lab04.e2.png", gauss_deviation);

    cv::waitKey(0);
}
