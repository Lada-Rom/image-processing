## Работа 3. Яркостные преобразования
автор: Толстенко Л. С.
дата: 02.03.2021

<https://mysvn.ru/LS/tolstenko_l_s/prj.labs/lab03/>

### Задание
1. В качестве тестового использовать изображение data/cross_0256x0256.png
2. Сгенерировать нетривиальную новую функцию преобразования яркости (не стоит использовать линейную функцию, гамму, случайная).
3. Сгенерировать визуализацию функцию преобразования яркости в виде изображения размером 512x512, черные точки а белом фоне.
4. Преобразовать пиксели grayscale версии тестового изображения при помощи LUT для сгенерированной функции преобразования.
4. Преобразовать пиксели каждого канала тестового изображения при помощи LUT для сгенерированной функции преобразования.
5. Результы сохранить для вставки в отчет.

### Результаты

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab03_rgb.png)
Рис. 1. Исходное тестовое изображение

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab03_gre.png)
Рис. 2. Тестовое изображение greyscale

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab03_gre_res.png)
Рис. 3. Результат применения функции преобразования яркости для greyscale

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab03_rgb_res.png)
Рис. 4. Результат применения функции преобразования яркости для каналов

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab03_viz_func.png)
Рис. 5. Визуализация функции яркостного преобразования

### Текст программы

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

const size_t WIDTH = 512;
const size_t HEIGHT = 512;
const size_t GRADATION = 256;

void showWindow(const cv::Mat& image, std::string name = "Image",
    size_t x = 0, size_t y = 0) {
    imshow(name, image);
    cv::moveWindow(name, x, y);
}

double brightnessConvertion(double x) {
    //return std::abs(x * std::cos(x));
    //return -1 / 35665 * x*x*x + 0.007 * x*x + 10;
    return 1 / 5.7 * std::abs(-0.001 * x*x*x + 0.3 * x*x - 8 * x - 1000);
}

void LUTConversion(const cv::Mat& source, cv::Mat& dest) {
    const size_t lut_width = 256;
    const size_t lut_height = 1;
    cv::Mat lut = cv::Mat::zeros(lut_height, lut_width, CV_8UC1);

    for (size_t i(0); i < lut_width; ++i)
        lut.at<uchar>(cv::Point(i, 0)) = brightnessConvertion(i);
    //showWindow(lut, "LUT", 0, 0);
    cv::LUT(source, lut, dest);
}

void channelsLUTConversion(const cv::Mat& source, cv::Mat& dest) {
    std::vector<cv::Mat> source_channels;
    cv::split(source, source_channels);

    for (cv::Mat& e : source_channels)
        LUTConversion(e, e);
    cv::Mat source_image;
    cv::merge(source_channels, source_image);

    cv::Mat red     = cv::Mat::zeros(source.rows, source.cols, CV_8UC3);
    cv::Mat green   = cv::Mat::zeros(source.rows, source.cols, CV_8UC3);
    cv::Mat blue    = cv::Mat::zeros(source.rows, source.cols, CV_8UC3);

    cv::Mat channels[] = { blue, green, red };
    int from_to[] = { 0, 0, 1, 4, 2, 8 };
    cv::mixChannels(&source_image, 1, channels, 3, from_to, 3);

    red.copyTo(dest(cv::Rect(   0, 0, source.cols, source.rows)));
    green.copyTo(dest(cv::Rect( source.cols, 0, source.cols, source.rows)));
    blue.copyTo(dest(cv::Rect(  source.cols * 2, 0, source.cols, source.rows)));
}

void drawFunction(cv::Mat& plot) {
    double j;
    for (size_t i(0); i < plot.cols; i += 2) {
        j = brightnessConvertion(i / 2);
        plot.at<uchar>(cv::Point(i, plot.rows - j - 1)) = 0;
        j = brightnessConvertion(i / 2 + 1);
        plot.at<uchar>(cv::Point(i + 1, plot.rows - j - 1)) = 0;
    }
}

int main() {
    cv::Mat source_image = cv::imread(
        "../../../data/data_cross_0256x0256.png");
    cv::Mat gray_image;
    cv::Mat converted_image;
    cv::Mat converted_channels = cv::Mat::zeros(
        source_image.rows, 3 * source_image.cols, CV_8UC3);
    cv::Mat function_plot = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);

    cv::cvtColor(source_image, gray_image, cv::COLOR_BGR2GRAY);
    function_plot += 255;
    drawFunction(function_plot);
    LUTConversion(gray_image, converted_image);
    channelsLUTConversion(source_image, converted_channels);

    showWindow(source_image, "Source image", 0, 0);
    showWindow(gray_image, "Gray image", 0, source_image.rows + 35);
    showWindow(converted_image, "Converted", 0, source_image.rows * 2 + 35 * 2);
    showWindow(function_plot, "Logarithm", source_image.cols + 10, 0 + 35);
    showWindow(converted_channels, "Converted channels",
        source_image.cols + 10, function_plot.rows + 35 * 2);

    cv::imwrite("../../../data/lab03_rgb.png", source_image);
    cv::imwrite("../../../data/lab03_gre.png", gray_image);
    cv::imwrite("../../../data/lab03_gre_res.png", converted_image);
    cv::imwrite("../../../data/lab03_rgb_res.png", converted_channels);
    cv::imwrite("../../../data/lab03_viz_func.png", function_plot);

    cv::waitKey(0);
}

```
