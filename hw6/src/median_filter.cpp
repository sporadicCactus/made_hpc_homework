#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <median_filter.h>

using namespace cv;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: median_filter <input_file> <output_file> <filter_height> <filter_width>" << "\n";
        return 1;
    }
    int fil_H = atoi(argv[3]);
    int fil_W = atoi(argv[4]);

    Mat image = imread(argv[1], cv::IMREAD_COLOR);
    
    Mat new_image = Mat::zeros(Size(image.size[0]-fil_H+1, image.size[1]-fil_W+1), CV_8UC3);

    apply_median_filter(image.data, new_image.data, image.size[0], image.size[1], 3, fil_H, fil_W, 8);

    imwrite(argv[2], new_image);
    return 0;
}
