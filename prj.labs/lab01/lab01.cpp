#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

const double gamma = 2.2;
const double num = 10;
const size_t WIDTH = 768;
const size_t HEIGHT = 60;

void powByMethod(cv::Mat& image) {
    cv::Rect rect1 = cv::Rect(0, 0, WIDTH, HEIGHT);
    cv::Rect rect2 = cv::Rect(0, HEIGHT, WIDTH, HEIGHT);
    cv::Mat tmp1(image, rect1);
    cv::Mat tmp2(image, rect2);
    tmp1.convertTo(tmp1, CV_32F);
    cv::pow(tmp1 / 255, 1.0 / gamma, tmp2);
    tmp2 *= 255;
    tmp2.copyTo(image(rect2));
}

void powByPixel(cv::Mat& image) {
    for (size_t i(0); i < WIDTH; ++i) {
        for (size_t j(0); j < HEIGHT; ++j) {
            uchar color = image.at<uchar>(cv::Point(i, j));
            color = std::pow((double)color / 255, 1 / gamma) * 255;
            image.at<uchar>(cv::Point(i, j + 2 * HEIGHT)) = color;
        }
    }
}

void showValues(const std::vector<std::chrono::duration<double>>& vec) {
    for (const auto& e : vec)
        std::cout << e.count() * 1000 << " ";
    std::cout << std::endl;
}

double countAvg(const std::vector<std::chrono::duration<double>>& vec) {
    double avg = 0;
    for (const auto& e : vec)
        avg += e.count() * 1000;
    avg /= num;
    return avg;
}

double countDisp(const std::vector<std::chrono::duration<double>>& vec) {
    double disp = 0;
    double avg = countAvg(vec);
    for (const auto& e : vec)
        disp += (e.count() * 1000 - avg) * (e.count() * 1000 - avg);
    disp /= num;
    return disp;
}

int main() {

    cv::Mat image(cv::Mat::zeros(3 * HEIGHT, WIDTH, CV_8UC1));

    for (size_t i(0); i < WIDTH; i += 3) {
        cv::line(image, cv::Point(i, 0), cv::Point(i, HEIGHT), i / 3);
        cv::line(image, cv::Point(i + 1, 0), cv::Point(i + 1, HEIGHT), i / 3);
        cv::line(image, cv::Point(i + 2, 0), cv::Point(i + 2, HEIGHT), i / 3);
    }

    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;
    std::chrono::duration<double> duration;
    std::vector<std::chrono::duration<double>> method_time;
    std::vector<std::chrono::duration<double>> pixel_time;

    for (size_t i(0); i < num; ++i) {
        start = std::chrono::system_clock::now();
        powByMethod(image);
        end = std::chrono::system_clock::now();
        duration = end - start;
        method_time.push_back(duration);
    }

    for (size_t i(0); i < num; ++i) {
        start = std::chrono::system_clock::now();
        powByPixel(image);
        end = std::chrono::system_clock::now();
        duration = end - start;
        pixel_time.push_back(duration);
    }

    std::cout << "Method: ";
    showValues(method_time);
    std::cout << "Average: " << countAvg(method_time) << std::endl;
    std::cout << "Dispersion: " << countDisp(method_time) << std::endl;

    std::cout << std::endl;

    std::cout << "Pixel: ";
    showValues(pixel_time);
    std::cout << "Average: " << countAvg(pixel_time) << std::endl;
    std::cout << "Dispersion: " << countDisp(pixel_time) << std::endl;

    cv::imwrite("../../../data/lab01.png", image); 

    imshow("Image", image);
    cv::waitKey(0);
}
