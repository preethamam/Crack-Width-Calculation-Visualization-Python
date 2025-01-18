/*
C code translation and implementation of Skeletonization
------------------------------------------------------------

Written by Dr. Preetham Manjunatha
Packaged December 2024

This package comes with no warranty of any kind (see below).


Description
-----------

The files in this package comprise the C implementation of a
AFMM method for skeletonizing binary images.  

Article reference: 
1. Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching 
method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. 
University of Groningen, Johann Bernoulli Institute for Mathematics and
Computer Science, 2002.
2. Reniers, Dennie & Telea, Alexandru. (2007). Tolerance-Based Feature 
Transforms. 10.1007/978-3-540-75274-5_12.


This implementation of a skeletonization method is the Go code translation and 
optimization to C code originally written by João Rafael Diniz Ramos's. 
Weblink: https://github.com/Joao-R/afmm

The code is written in C and is intended to be compiled with a C compiler.

Alex (article author) has a faster C/C++ implementation. It can be found at:

1. https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/
2. https://webspace.science.uu.nl/~telea001/Software/Software.

Execute main.c | ./afmm.exe input.jpg 100 1 <input_image> <threshold> <is_rgb>


Copyright Notice
----------------
This software comes with no warranty, expressed or implied, including
but not limited to merchantability or fitness for any particular
purpose.

All files are copyright Dr. Preetham Manjunatha.  
Permission is granted to use the material for noncommercial and 
research purposes.
*/

#ifndef AFMM_H
#define AFMM_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

// OS-specific threading includes
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else
    #include <pthread.h>
#endif

// C++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

// Data structures

// Image structure to hold image data
typedef struct {
    uint32_t width;  // Image width
    uint32_t height; // Image height
    uint32_t* data;  // RGBA data
} Image;

// DataGrid structure to hold grid data for AFMM
typedef struct {
    int* U;         // Distance values
    double* T;      // Pixel miss values
    uint8_t* f;     // Pixel hit values
    
    int* set;       // Set identifiers
    double* x;      // X coordinates
    double* y;      // Y coordinates
    
    int colNum;     // Number of columns
    int rowNum;     // Number of rows
} DataGrid;

// PixelHeap structure for heap operations
typedef struct {
    DataGrid* data; // Pointer to DataGrid
    int* heapIndex; // Heap indices
    int* pixelIds;  // Pixel IDs
    int size;       // Current size of the heap
    int capacity;   // Capacity of the heap
} PixelHeap;

// Thread data structure
typedef struct {
    DataGrid* state;    // Pointer to DataGrid
    bool startInFront;  // Flag to indicate start position
} ThreadData;

// Function declarations

// Core AFMM functions
double* FMM(const Image* img, int is_rgb);
void AFMM(const Image* img, int is_rgb, double** deltaU_out, double** DT_out);
void Skeletonize(const Image* img, double threshold, int is_rgb, 
                 uint8_t** output_skeleton, double** output_deltaU, double** output_DT);

// Core algorithm function declarations
void init_FMM(DataGrid* state, PixelHeap* band);
void init_AFMM(DataGrid* state, PixelHeap* band, bool startInFront);
void step_FMM(DataGrid* d, PixelHeap* band);
void step_AFMM(DataGrid* d, PixelHeap* band);
void afmm(DataGrid* state, bool startInFront);

// Image handling functions
Image* load_image(const char* filename, int is_rgb);
void save_distance_transform(const double* DT, int width, int height, const char* filename);
void save_deltaU(const double* deltaU, int width, int height, const char* filename);
void save_skeleton(const uint8_t* skeleton, int width, int height, const char* filename);
void cleanup_image(Image* img);

// Timing function
void print_elapsed_time(clock_t start_time, clock_t end_time, const char* operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // AFMM_H