## CUDA homework

OpenCV is used to read and write images.
$CPATH should contain the path to `opencv2` dir with headers.
$LIBRARY_PATH should contain the path to a directory with OpenCV `.so` files.

Usage:
```
make

./linear_filter <input_file> <output_file> <filter_file>
./median_filter <input_file> <filter_height> <filter_width>
./histogram <input_file>
```
