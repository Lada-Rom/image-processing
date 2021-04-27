## Работа 1. Исследование гамма-коррекции

автор: Толстенко Л. С.  
дата: 25.02.2021

<https://mysvn.ru/LS/tolstenko_l_s/prj.labs/lab01/>

### Задание

1. Сгенерировать серое тестовое изображение $I_1$ в виде прямоугольника размером 768х60 пикселя с плавным изменение пикселей от черного к белому, одна градация серого занимает 3 пикселя по горизонтали.
2. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_1$ при помощи функци pow.
3. Применить  к изображению $I_1$ гамма-коррекцию с коэффициентом из интервала 2.2-2.4 и получить изображение $G_2$ при помощи прямого обращения к пикселям.
4. Показать визуализацию результатов в виде одного изображения (сверху вниз $I_1$, $G_1$, $G_2$).
5. Сделать замер времени обработки изображений в п.2 и п.3, результаты отфиксировать в отчете.

### Результаты

![](https://mysvn.ru/LS/tolstenko_l_s/data/lab01.png)  
Рис. 1. Результаты работы программы (сверху вниз $I_1$, $G_1$, $G_2$)

Количество испытаний: 10

Использование cv::pow (мс):  
3.3871 3.3829 3.5844 3.6851 2.7489 2.9193 3.3452 3.555 3.6254 2.6854  
Матожидание: 3.29187 мс  
Дисперсия: 0.124231 мс

Прямое обращение к пикселям (мс):  
6.2988 6.7447 6.2493 6.7653 6.6501 6.2431 6.6947 6.3246 6.1413 6.649  
Матожидание: 6.47609 мс  
Дисперсия: 0.0535945 мс

### Текст программы

```cpp
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
```