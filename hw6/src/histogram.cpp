#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <histogram.h>

using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: histogram <input_file>" << "\n"; 
        return 1;
    }
    
    Mat image = imread(argv[1], cv::IMREAD_COLOR);

    int hist[256];

    histogram(image.data, hist, image.size[0], image.size[1], 128);

    for (int i = 0; i < 255; i++) {
        std::cout << hist[i] << " ";
    }
    std::cout << hist[255] << "\n";

    return 0;
}
