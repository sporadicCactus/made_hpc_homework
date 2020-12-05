#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <linear_filter.h>

using namespace cv;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: linear_filter in_file out_file filter" << "\n";
        return 1;
    }

    Mat image = imread(argv[1], cv::IMREAD_COLOR);
    
    std::ifstream filter_file(argv[3]);
    int fil_H, fil_W;
    filter_file >> fil_H;
    filter_file >> fil_W;
    float *filter = (float*)malloc(sizeof(float)*fil_H*fil_W);

    for (int i = 0; i < fil_H; i++) {
        for (int j = 0; j < fil_W; j++) {
            filter_file >> filter[i*fil_W + j];
        }
    }
    filter_file.close();

    Mat new_image = Mat::zeros(Size(image.size[0]-fil_H+1, image.size[1]-fil_W+1), CV_8UC3);

    apply_linear_filter(image.data, new_image.data, filter, image.size[0], image.size[1], 3, fil_H, fil_W, 8);

    imwrite(argv[2], new_image);
    return 0;
}
