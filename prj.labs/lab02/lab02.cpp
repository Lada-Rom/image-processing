#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void showWindow(const cv::Mat& image, std::string name = "Image",
    size_t x = 0, size_t y = 0) {
    imshow(name, image);
    cv::moveWindow(name, x, y);
}

void compressImage(const cv::Mat& image, int degree) {
    cv::imwrite("../../../data/cross_0256x0256_025.jpg",
        image, {cv::IMWRITE_JPEG_QUALITY, degree});
}
cv::Mat makeHistogram(const cv::Mat& image, const cv::Scalar& color,
    size_t width, size_t height, size_t gradation) {

    cv::Mat plot(cv::Mat::zeros(height, width, CV_8UC3));;
    plot += cv::Scalar(255, 255, 255);

    std::vector<size_t> levels(gradation);
    for (size_t i(0); i < image.cols; ++i)
        for (size_t j(0); j < image.rows; ++j)
            ++levels[image.at<uchar>(cv::Point(i, j))];

    cv::Rect rect;
    double max = *std::max_element(levels.begin(), levels.end());
    for (size_t i(0); i < gradation; ++i) {
        rect = cv::Rect(2 * i, height * (1 - levels[i] / max), 2, levels[i]);
        cv::rectangle(plot, rect, color, -1);
    }

    return plot;
}
void fillWithChannels(const cv::Mat& image, cv::Mat& mat) {
    cv::Mat red     = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::Mat green   = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::Mat blue    = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);

    cv::Mat channels[] = { blue, green, red };
    int from_to[] = { 0, 0, 1, 4, 2, 8 };
    cv::mixChannels(&image, 1, channels, 3, from_to, 3);

    image.copyTo(mat(cv::Rect(0, 0, image.cols, image.rows)));
    red.copyTo(mat(cv::Rect(image.cols, 0, image.cols, image.rows)));
    green.copyTo(mat(cv::Rect(0, image.rows, image.cols, image.rows)));
    blue.copyTo(mat(cv::Rect(image.cols, image.rows, image.cols, image.rows)));
}

void showChannelsGray(const cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    showWindow(channels[0], "Blue channel", image.cols, 0);
    showWindow(channels[1], "Green channel", 0, image.rows + 35);
    showWindow(channels[2], "Red channel", image.cols, image.rows + 35);
}
cv::Mat makeChannelsHistogram(
    const cv::Mat& image, size_t width, size_t height, size_t gradation) {

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);
    
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat plot = cv::Mat::zeros(2 * height, 2 * width, CV_8UC3);
    plot += cv::Scalar(255, 255, 255);

    cv::Rect rect1 = cv::Rect(0, 0, width, height);
    cv::Rect rect2 = cv::Rect(width, 0, width, height);
    cv::Rect rect3 = cv::Rect(0, height, width, height);
    cv::Rect rect4 = cv::Rect(width, height, width, height);

    cv::Mat mat1(plot, rect1);     cv::Mat mat2(plot, rect2);
    cv::Mat mat3(plot, rect3);     cv::Mat mat4(plot, rect4);

    mat1 = makeHistogram(
        gray_image, cv::Scalar({0, 0, 0}), width, height, gradation);
    mat2 = makeHistogram(
        channels[0], cv::Scalar({ 0, 0, 255 }), width, height, gradation);
    mat3 = makeHistogram(
        channels[1], cv::Scalar({ 0, 255, 0 }), width, height, gradation);
    mat4 = makeHistogram(
        channels[2], cv::Scalar({ 255, 0, 0 }), width, height, gradation);

    mat1.copyTo(plot(rect1));
    mat2.copyTo(plot(rect2));
    mat3.copyTo(plot(rect3));
    mat4.copyTo(plot(rect4));

    return plot;
}

int main() {
    const size_t width = 512;
    const size_t height = 256;
    const size_t gradation = 256;

    cv::Mat source_image = cv::imread(
        "../../../data/data_cross_0256x0256.png");
    compressImage(source_image, 25);
    cv::Mat compressed_image = cv::imread(
        "../../../data/cross_0256x0256_025.jpg");

    cv::Mat source_channels = cv::Mat::zeros(
        2 * source_image.rows, 2 * source_image.cols, CV_8UC3);
    cv::Mat compressed_channels = cv::Mat::zeros(
        2 * source_image.rows, 2 * source_image.cols, CV_8UC3);

    fillWithChannels(source_image, source_channels);
    fillWithChannels(compressed_image, compressed_channels);

    cv::imwrite("../../../data/cross_0256x0256_025.jpg", compressed_image);
    cv::imwrite(
        "../../../data/cross_0256x0256_png_channels.png", source_channels);
    cv::imwrite(
        "../../../data/cross_0256x0256_jpg_channels.png", compressed_channels);

    showWindow(source_channels, "Source Image", 0, 0);
    showWindow(compressed_channels,
        "Compressed Image", source_channels.cols + 2, 0);

    cv::Mat source_plot = makeChannelsHistogram(
        source_image, width, height, gradation);
    cv::Mat compressed_plot = makeChannelsHistogram(
        compressed_image, width, height, gradation);

    cv::imwrite("../../../data/cross_0256x0256_hist1.png", source_plot);
    cv::imwrite("../../../data/cross_0256x0256_hist2.png", compressed_plot);

    showWindow(source_plot, "Source plot", source_plot.cols, 0);
    showWindow(compressed_plot, "Compressed plot",
        compressed_plot.cols, compressed_plot.rows);

    cv::waitKey(0);
}
