# AFMM Skeletonization

This repository contains the C implementation of the Augmented Fast Marching Method (AFMM) for skeletonizing binary images. The implementation is based on the work by Dr. Preetham Manjunatha and includes Python bindings for easy integration with Python projects.

## Description

The AFMM method is used for computing skeletons and centerlines of binary images. This implementation is a translation and optimization of the original Go code written by João Rafael Diniz Ramos.

### Article References
1. Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, 2002.
2. Reniers, Dennie & Telea, Alexandru. (2007). Tolerance-Based Feature Transforms. 10.1007/978-3-540-75274-5_12.

## Repository Structure

```
afmm.c
afmm.h
pybind_afmm.cpp
Readme.md
setup_command.txt
setup.py
stb_image_write.h
stb_image.h
```

## Requirements

- Python 3.6 or higher
- C/C++ compiler
    - GCC (MinGW for Windows) or 
    - MSYS2 (for Windows) or
    - MSVC/Visual Studio Build Tools (for Windows)
- [pybind11](https://github.com/pybind/pybind11) >= 2.5.0
- [numpy](https://numpy.org/) >= 1.13.0
- [Pillow](https://python-pillow.org/)

## Installation

1. Change directory:
    ```sh
    cd pyafmm
    ```

2. Build the Cython extension in place:
    ```sh
    python setup.py build_ext --inplace
    ```

3. Install the package:
    ```sh
    pip install -e .
    ```

## Usage

### Command Line

You can execute the AFMM skeletonization from the command line using the provided executable (C only after building executable):

```sh
./afmm.exe input.jpg 100 1 <input_image> <threshold> <is_rgb>
```
- `<input_image>`: Path to the input image file.
- `<threshold>`: Threshold value for skeletonization.
- `<is_rgb>`: Set to `1` if the input image is RGB channels (binary image with three channels), otherwise set to `0`.

### Python

You can also use the Python bindings to integrate AFMM skeletonization into your Python projects:

```python
import pyafmm
import numpy as np
from PIL import Image

# Load an image
image = Image.open('input.jpg').convert('L')
image_data = np.array(image)

# Perform skeletonization
skeleton = pyafmm.skeletonize(image_data, threshold=100, is_rgb=False)

# Save the result
result_image = Image.fromarray(skeleton)
result_image.save('output.png')
```

## Functions

### C Functions

- `void save_distance_transform(const double* DT, int width, int height, const char* filename);`
- `void save_deltaU(const double* deltaU, int width, int height, const char* filename);`
- `void save_skeleton(const uint8_t* skeleton, int width, int height, const char* filename);`
- `void cleanup_image(Image* img);`
- `void print_elapsed_time(clock_t start_time, clock_t end_time, const char* operation);`

### Python Functions

- `pyafmm.skeletonize(image_data, threshold, is_rgb)`

## License

This software comes with no warranty, expressed or implied, including but not limited to merchantability or fitness for any particular purpose. All files are copyright Dr. Preetham Manjunatha. Permission is granted to use the material for noncommercial and research purposes.

## References

- [AFMM Software by Alex Telea](https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/)
- [Original Go implementation by João Rafael Diniz Ramos](https://github.com/Joao-R/afmm)
