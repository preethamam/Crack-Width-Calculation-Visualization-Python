#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "afmm.h"

namespace py = pybind11;

// Helper function to convert numpy array to Image struct
Image* numpy_to_image(py::array_t<uint8_t>& input_array, bool is_rgb) {
    py::buffer_info buf = input_array.request();
    
    if (is_rgb && buf.ndim != 3) {
        throw std::runtime_error("RGB mode requires a 3D input array");
    }
    if (!is_rgb && buf.ndim != 2) {
        throw std::runtime_error("Grayscale mode requires a 2D input array");
    }
    
    Image* img = new Image;
    img->height = buf.shape[0];
    img->width = buf.shape[1];
    img->data = new uint32_t[img->width * img->height];
    
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    
    if (!is_rgb) {  // Grayscale
        for (size_t i = 0; i < img->width * img->height; i++) {
            img->data[i] = ptr[i] > 0 ? 1 : 0;
        }
    } else {  // RGB
        if (buf.shape[2] != 3) {
            delete[] img->data;
            delete img;
            throw std::runtime_error("RGB image must have 3 channels");
        }
        for (size_t i = 0; i < img->width * img->height; i++) {
            uint8_t r = ptr[i * 3];
            uint8_t g = ptr[i * 3 + 1];
            uint8_t b = ptr[i * 3 + 2];
            img->data[i] = (r << 24) | (g << 16) | (b << 8) | 0xFF;
        }
    }
    
    return img;
}

py::array_t<double> fmm_wrapper(py::array_t<uint8_t> input_array, bool is_rgb) {
    Image* img = numpy_to_image(input_array, is_rgb);
    
    // Call FMM function
    double* DT = FMM(img, is_rgb);
    
    // Create numpy array for output
    py::array_t<double> output_array({img->height, img->width});
    
    // Copy data to numpy array
    std::memcpy(output_array.request().ptr, DT,
                img->width * img->height * sizeof(double));
    
    // Cleanup
    delete[] img->data;
    delete img;
    free(DT);
    
    return output_array;
}

py::tuple afmm_wrapper(py::array_t<uint8_t> input_array, bool is_rgb) {
    Image* img = numpy_to_image(input_array, is_rgb);
    
    // Output pointers
    double* deltaU = nullptr;
    double* DT = nullptr;
    
    // Call AFMM function
    AFMM(img, is_rgb, &deltaU, &DT);
    
    // Create numpy arrays for outputs
    py::array_t<double> deltaU_array({img->height, img->width});
    py::array_t<double> DT_array({img->height, img->width});
    
    // Copy data to numpy arrays
    std::memcpy(deltaU_array.request().ptr, deltaU,
                img->width * img->height * sizeof(double));
    std::memcpy(DT_array.request().ptr, DT,
                img->width * img->height * sizeof(double));
    
    // Cleanup
    delete[] img->data;
    delete img;
    free(deltaU);
    free(DT);
    
    return py::make_tuple(deltaU_array, DT_array);
}

py::tuple skeletonize_wrapper(py::array_t<uint8_t> input_array, double threshold, bool is_rgb) {
    Image* img = numpy_to_image(input_array, is_rgb);
    
    // Output pointers
    uint8_t* skeleton = nullptr;
    double* deltaU = nullptr;
    double* DT = nullptr;
    
    // Call Skeletonize function
    Skeletonize(img, threshold, is_rgb, &skeleton, &deltaU, &DT);
    
    // Create numpy arrays for outputs
    py::array_t<uint8_t> skeleton_array({img->height, img->width});
    py::array_t<double> deltaU_array({img->height, img->width});
    py::array_t<double> DT_array({img->height, img->width});
    
    // Copy data to numpy arrays
    std::memcpy(skeleton_array.request().ptr, skeleton, 
                img->width * img->height * sizeof(uint8_t));
    std::memcpy(deltaU_array.request().ptr, deltaU, 
                img->width * img->height * sizeof(double));
    std::memcpy(DT_array.request().ptr, DT, 
                img->width * img->height * sizeof(double));
    
    // Cleanup
    delete[] img->data;
    delete img;
    free(skeleton);
    free(deltaU);
    free(DT);
    
    return py::make_tuple(skeleton_array, deltaU_array, DT_array);
}

PYBIND11_MODULE(pyafmm, m) {
    m.doc() = "Python binding for Augmented Fast Marching Method Skeletonization";
    
    m.def("fmm", &fmm_wrapper,
          py::arg("image"),
          py::arg("is_rgb") = false,
          "Compute the Fast Marching Method distance transform.\n\n"
          "Parameters:\n"
          "    image (numpy.ndarray): Input image (2D grayscale or 3D RGB)\n"
          "    is_rgb (bool): Whether the input image is RGB\n\n"
          "Returns:\n"
          "    numpy.ndarray: Distance transform of the input");
          
    m.def("afmm", &afmm_wrapper,
          py::arg("image"),
          py::arg("is_rgb") = false,
          "Compute the Augmented Fast Marching Method transform.\n\n"
          "Parameters:\n"
          "    image (numpy.ndarray): Input image (2D grayscale or 3D RGB)\n"
          "    is_rgb (bool): Whether the input image is RGB\n\n"
          "Returns:\n"
          "    tuple: (deltaU, distance_transform)\n"
          "        deltaU: Skeleton measure values\n"
          "        distance_transform: Distance transform of the input");
    
    m.def("skeletonize", &skeletonize_wrapper,
          py::arg("image"),
          py::arg("threshold") = 100.0,
          py::arg("is_rgb") = false,
          "Compute the skeleton of a binary image using AFMM.\n\n"
          "Parameters:\n"
          "    image (numpy.ndarray): Input image (2D grayscale or 3D RGB)\n"
          "    threshold (float): Threshold value for skeleton detection\n"
          "    is_rgb (bool): Whether the input image is RGB\n\n"
          "Returns:\n"
          "    tuple: (skeleton, deltaU, distance_transform)\n"
          "        skeleton: Binary image of the skeleton\n"
          "        deltaU: Skeleton measure values\n"
          "        distance_transform: Distance transform of the input");
}