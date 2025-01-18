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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "afmm.h"

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else
    #include <pthread.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Linked list implementation for AFMM initialization
typedef struct ListNode {
    int value;
    struct ListNode* next;
    struct ListNode* prev;
} ListNode;

typedef struct List {
    ListNode* head;
    ListNode* tail;
    int size;
} List;

// Create a new list
static List* list_create() {
    List* list = malloc(sizeof(List));
    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    return list;
}

// Push a value to the front of the list
static ListNode* list_push_front(List* list, int value) {
    ListNode* node = malloc(sizeof(ListNode));
    node->value = value;
    node->prev = NULL;
    node->next = list->head;

    if (list->head) {
        list->head->prev = node;
    } else {
        list->tail = node;
    }

    list->head = node;
    list->size++;
    return node;
}

// Push a value to the back of the list
static ListNode* list_push_back(List* list, int value) {
    ListNode* node = malloc(sizeof(ListNode));
    node->value = value;
    node->next = NULL;
    node->prev = list->tail;

    if (list->tail) {
        list->tail->next = node;
    } else {
        list->head = node;
    }

    list->tail = node;
    list->size++;
    return node;
}

// Remove the front node from the list
static int list_remove_front(List* list) {
    if (!list->head) return -1;

    ListNode* node = list->head;
    int value = node->value;

    list->head = node->next;
    if (list->head) {
        list->head->prev = NULL;
    } else {
        list->tail = NULL;
    }

    free(node);
    list->size--;
    return value;
}

// Remove a specific node from the list
static void list_remove_node(List* list, ListNode* node) {
    if (!node) return;

    if (node->prev) {
        node->prev->next = node->next;
    } else {
        list->head = node->next;
    }

    if (node->next) {
        node->next->prev = node->prev;
    } else {
        list->tail = node->prev;
    }

    free(node);
    list->size--;
}

// Destroy the list and free memory
static void list_destroy(List* list) {
    ListNode* current = list->head;
    while (current) {
        ListNode* next = current->next;
        free(current);
        current = next;
    }
    free(list);
}

// Heap functions for AFMM
static int heap_parent(int i) {
    return (i - 1) / 2;
}

static int heap_left_child(int i) {
    return 2 * i + 1;
}

static int heap_right_child(int i) {
    return 2 * i + 2;
}

static bool heap_less(PixelHeap* h, int i, int j) {
    return h->data->T[h->pixelIds[i]] < h->data->T[h->pixelIds[j]];
}

static void heap_swap(PixelHeap* h, int i, int j) {
    int temp = h->pixelIds[i];
    h->pixelIds[i] = h->pixelIds[j];
    h->pixelIds[j] = temp;
    h->heapIndex[h->pixelIds[i]] = i;
    h->heapIndex[h->pixelIds[j]] = j;
}

// Sift up operation for heap
static void heap_sift_up(PixelHeap* h, int i) {
    while (i > 0 && heap_less(h, i, heap_parent(i))) {
        heap_swap(h, i, heap_parent(i));
        i = heap_parent(i);
    }
}

// Sift down operation for heap
static void heap_sift_down(PixelHeap* h, int i) {
    int min_index = i;
    int left = heap_left_child(i);
    int right = heap_right_child(i);
    
    if (left < h->size && heap_less(h, left, min_index)) {
        min_index = left;
    }
    if (right < h->size && heap_less(h, right, min_index)) {
        min_index = right;
    }
    
    if (i != min_index) {
        heap_swap(h, i, min_index);
        heap_sift_down(h, min_index);
    }
}

// Push a value to the heap
static void heap_push(PixelHeap* h, int x) {
    if (h->data->f[x] == 1) {
        h->heapIndex[x] = h->size;
        h->pixelIds[h->size] = x;
        h->size++;
        heap_sift_up(h, h->size - 1);
        return;
    }
    
    heap_sift_up(h, h->heapIndex[x]);
    heap_sift_down(h, h->heapIndex[x]);
}

// Pop the minimum value from the heap
static int heap_pop(PixelHeap* h) {
    int result = h->pixelIds[0];
    h->size--;
    if (h->size > 0) {
        h->pixelIds[0] = h->pixelIds[h->size];
        h->heapIndex[h->pixelIds[0]] = 0;
        heap_sift_down(h, 0);
    }
    return result;
}

// Initialize the heap
static void heap_init(PixelHeap* h) {
    // Initialize heap size
    if (!h) return;
    
    // Build heap property from bottom up
    for (int i = h->size / 2 - 1; i >= 0; i--) {
        heap_sift_down(h, i);
    }
}

// Image parsing function
static void parse_image(DataGrid* imgData, const Image* img, int rgb_binary) {
    imgData->colNum = img->width + 2;
    imgData->rowNum = img->height + 2;
    
    imgData->f = calloc(imgData->colNum * imgData->rowNum, sizeof(uint8_t));
    imgData->T = calloc(imgData->colNum * imgData->rowNum, sizeof(double));
    
    int xx, yy;  // relative x and y
    yy = 1;
    for (uint32_t y = 0; y < img->height; y++) {
        xx = 1;
        for (uint32_t x = 0; x < img->width; x++) {
            uint32_t pixel = img->data[y * img->width + x];
            if (rgb_binary) {
                uint32_t r = ((pixel >> 24) & 0xFF) << 8;
                uint32_t g = ((pixel >> 16) & 0xFF) << 8;
                uint32_t b = ((pixel >> 8) & 0xFF) << 8;
                
                double lum = 0.299 * r + 0.587 * g + 0.114 * b;
                
                int idx = xx+yy*imgData->colNum;
                if (lum > 32768) {
                    imgData->f[idx] = 1;
                    imgData->T[idx] = INFINITY;
                } else {
                    imgData->f[idx] = 0;
                    imgData->T[idx] = 0;
                }
            } else {
                int idx = xx+yy*imgData->colNum;
                if (pixel > 0) {
                    imgData->f[idx] = 1;
                    imgData->T[idx] = INFINITY;
                } else {
                    imgData->f[idx] = 0;
                    imgData->T[idx] = 0;
                }
            }
            xx++;
        }
        yy++;
    }
}

// Neighborhood functions implementation
static void von_neumann_neighborhood(int idx, int colNum, int* result) {
    int x = idx % colNum;
    int y = idx / colNum;
    y = y * colNum;
    
    result[0] = y + x - 1;
    result[1] = y - colNum + x;
    result[2] = y + x + 1;
    result[3] = y + colNum + x;
}

static void moore_neighborhood(int idx, int colNum, int* result) {
    int x = idx % colNum;
    int y = idx / colNum;
    y = y * colNum;
    int ym1 = y - colNum;
    int yp1 = y + colNum;
    
    result[0] = ym1 + x - 1;
    result[1] = ym1 + x;
    result[2] = ym1 + x + 1;
    result[3] = y + x + 1;
    result[4] = yp1 + x + 1;
    result[5] = yp1 + x;
    result[6] = yp1 + x - 1;
    result[7] = y + x - 1;
}

// Safe Moore neighborhood function
static void safe_moore_neighborhood(DataGrid* d, int idx, int* result) {
    int neighbors[8];
    moore_neighborhood(idx, d->colNum, neighbors);
    int offset = 0;
    
    for (int i = 0; i < 8; i++) {
        if (d->f[neighbors[i]] == 0) {
            offset = i;
            break;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        result[i] = neighbors[(i + offset) % 8];
    }
}

// Initialization functions
void init_FMM(DataGrid* state, PixelHeap* band) {
    int idx;
    /* Boundary detection */
    band->data = state;
    band->heapIndex = calloc(state->rowNum * state->colNum, sizeof(int));
    band->pixelIds = calloc(state->rowNum * state->colNum, sizeof(int));
    band->size = 0;

    for (int y = 1; y < state->rowNum - 1; y++) {
        for (int x = 1; x < state->colNum - 1; x++) {
            idx = y * state->colNum + x;
            if (state->f[idx] == 1) {
                int neighbors[4];
                von_neumann_neighborhood(idx, state->colNum, neighbors);
                for (int i = 0; i < 4; i++) {
                    if (state->f[neighbors[i]] == 0) {
                        band->heapIndex[idx] = band->size;
                        band->pixelIds[band->size++] = idx;
                        state->T[idx] = 0;
                        state->f[idx] = 2;  // band
                        break;
                    }
                }
            }
        }
    }

    heap_init(band);
}

void init_AFMM(DataGrid* state, PixelHeap* band, bool startInFront) {
    int idx;
    List* bandList = list_create();
    
    // Create hash table (using array since we know the size)
    ListNode** idxToList = calloc(state->rowNum * state->colNum, sizeof(ListNode*));

    band->data = state;
    band->heapIndex = calloc(state->rowNum * state->colNum, sizeof(int));
    band->pixelIds = calloc(state->rowNum * state->colNum, sizeof(int));
    band->size = 0;

    // Boundary detection
    for (int y = 1; y < state->rowNum - 1; y++) {
        for (int x = 1; x < state->colNum - 1; x++) {
            idx = y * state->colNum + x;
            if (state->f[idx] == 1) {
                int neighbors[4];
                von_neumann_neighborhood(idx, state->colNum, neighbors);
                for (int i = 0; i < 4; i++) {
                    if (state->f[neighbors[i]] == 0) {
                        if (startInFront) {
                            idxToList[idx] = list_push_back(bandList, idx);
                        } else {
                            idxToList[idx] = list_push_front(bandList, idx);
                        }
                        band->pixelIds[band->size++] = idx;
                        state->T[idx] = 0;
                        state->f[idx] = 3;  // band uninitialized
                        break;
                    }
                }
            }
        }
    }

    // Initialize U
    state->x = calloc(band->size, sizeof(double));
    state->y = calloc(band->size, sizeof(double));
    state->set = calloc(band->size, sizeof(int));

    bool found;
    int current;
    int setID = 0;
    int count = 0;

    while (bandList->size > 0) {
        current = list_remove_front(bandList);

        state->U[current] = count;
        state->f[current] = 2;
        state->set[count] = setID;
        state->x[count] = current % state->colNum;
        state->y[count] = current / state->colNum;
        count++;

        // Propagation
        found = true;
        while (found) {
            found = false;
            int neighbors[8];
            safe_moore_neighborhood(state, current, neighbors);
            for (int i = 0; i < 8; i++) {
                int j = neighbors[i];
                if (state->f[j] == 3) {
                    current = j;
                    state->U[current] = count;
                    state->f[current] = 2;
                    state->set[count] = setID;
                    state->x[count] = current % state->colNum;
                    state->y[count] = current / state->colNum;
                    count++;
                    list_remove_node(bandList, idxToList[current]);

                    found = true;
                    break;
                }
            }
        }

        setID++;
    }

    heap_init(band);

    // Cleanup
    free(idxToList);
    list_destroy(bandList);
}

// Core algorithm functions
static void solve(int idx1, int idx2, double* T, uint8_t* f, double* solution) {
    double r, s;
    
    if (f[idx1] == 0) {
        if (f[idx2] == 0) {
            r = sqrt(2 - ((T[idx1] - T[idx2]) * (T[idx1] - T[idx2])));
            s = (T[idx1] + T[idx2] - r) * 0.5;
            if (s >= T[idx1] && s >= T[idx2]) {
                if (s < *solution) {
                    *solution = s;
                } else {
                    s += r;
                    if (s >= T[idx1] && s >= T[idx2]) {
                        if (s < *solution) {
                            *solution = s;
                        }
                    }
                }
            }
        } else {
            if (1 + T[idx1] < *solution) {
                *solution = 1 + T[idx1];
            }
        }
    } else {
        if (f[idx2] == 0) {
            if (1 + T[idx2] < *solution) {
                *solution = 1 + T[idx2];
            }
        }
    }
}

// Guess U values for AFMM
static void guess_U(DataGrid* d, int idx, int* neighbors) {
    double D = INFINITY;
    double distance;
    double dx, dy;
    
    for (int i = 0; i < 8; i++) {
        int neighbor = neighbors[i];
        if (d->f[neighbor] != 1) {
            dx = (idx % d->colNum) - d->x[d->U[neighbor]];
            dy = (idx / d->colNum) - d->y[d->U[neighbor]];
            distance = sqrt(dx*dx + dy*dy);
            
            if (distance < D) {
                D = distance;
                d->U[idx] = d->U[neighbor];
            }
        }
    }
}

// Step function for FMM
void step_AFMM(DataGrid* d, PixelHeap* band) {
    double solution;
    int current = heap_pop(band);
    int neighbors[4];
    int other_neighbors[4];
    
    d->f[current] = 0;
    von_neumann_neighborhood(current, d->colNum, neighbors);
    
    for (int i = 0; i < 4; i++) {
        int neighbor = neighbors[i];
        if (d->f[neighbor] != 0) {
            von_neumann_neighborhood(neighbor, d->colNum, other_neighbors);
            
            solution = d->T[neighbor];
            
            solve(other_neighbors[0], other_neighbors[1], d->T, d->f, &solution);
            solve(other_neighbors[2], other_neighbors[1], d->T, d->f, &solution);
            solve(other_neighbors[0], other_neighbors[3], d->T, d->f, &solution);
            solve(other_neighbors[2], other_neighbors[3], d->T, d->f, &solution);
            
            d->T[neighbor] = solution;
            heap_push(band, neighbor);
            
            if (d->f[neighbor] == 1) {
                d->f[neighbor] = 2;
                int moore_neighbors[8];
                moore_neighborhood(neighbor, d->colNum, moore_neighbors);
                guess_U(d, neighbor, moore_neighbors);
            }
        }
    }
}

// Step function for AFMM
void step_FMM(DataGrid* d, PixelHeap* band) {
    double solution;
    int current = heap_pop(band);
    int neighbors[4];
    int other_neighbors[4];
    
    d->f[current] = 0;
    von_neumann_neighborhood(current, d->colNum, neighbors);
    
    for (int i = 0; i < 4; i++) {
        int neighbor = neighbors[i];
        if (d->f[neighbor] != 0) {
            von_neumann_neighborhood(neighbor, d->colNum, other_neighbors);
            
            solution = d->T[neighbor];
            
            solve(other_neighbors[0], other_neighbors[1], d->T, d->f, &solution);
            solve(other_neighbors[2], other_neighbors[1], d->T, d->f, &solution);
            solve(other_neighbors[0], other_neighbors[3], d->T, d->f, &solution);
            solve(other_neighbors[2], other_neighbors[3], d->T, d->f, &solution);
            
            d->T[neighbor] = solution;
            heap_push(band, neighbor);
            
            if (d->f[neighbor] == 1) {
                d->f[neighbor] = 2;
            }
        }
    }
}

// Thread function declarations
#if defined(_WIN32) || defined(_WIN64)
    DWORD WINAPI afmm_thread(LPVOID arg) {
        ThreadData* data = (ThreadData*)arg;
        afmm(data->state, data->startInFront);
        return 0;
    }
#else
    void* afmm_thread(void* arg) {
        ThreadData* data = (ThreadData*)arg;
        afmm(data->state, data->startInFront);
        return NULL;
    }
#endif

// Core AFMM function
void afmm(DataGrid* state, bool startInFront) {
    PixelHeap band = {0};
    init_AFMM(state, &band, startInFront);
    
    while (band.size > 0) {
        step_AFMM(state, &band);
    }
    
    free(band.heapIndex);
    free(band.pixelIds);
}

// FMM function
double* FMM(const Image* img, int rgb_binary) {
    DataGrid state = {0};
    parse_image(&state, img, rgb_binary);

    PixelHeap band = {0};
    init_FMM(&state, &band);

    while (band.size > 0) {
        step_FMM(&state, &band);
    }

    double* DT = calloc((state.colNum - 2) * (state.rowNum - 2), sizeof(double));
    for (int y = 1; y < state.rowNum - 1; y++) {
        for (int x = 1; x < state.colNum - 1; x++) {
            int oldIdx = y * state.colNum + x;
            int newIdx = (y - 1) * (state.colNum - 2) + (x - 1);
            DT[newIdx] = state.T[oldIdx];
        }
    }

    free(band.heapIndex);
    free(band.pixelIds);
    free(state.T);
    free(state.f);

    return DT;
}

// AFMM function
void AFMM(const Image* img, int rgb_binary, double** deltaU_out, double** DT_out) {
    // Initialize stateFirst
    DataGrid stateFirst = {0};
    stateFirst.U = calloc((img->width + 2) * (img->height + 2), sizeof(int));
    parse_image(&stateFirst, img, rgb_binary);

    // Create copy of mask for later use
    uint8_t* mask = malloc(stateFirst.colNum * stateFirst.rowNum * sizeof(uint8_t));
    memcpy(mask, stateFirst.f, stateFirst.colNum * stateFirst.rowNum * sizeof(uint8_t));

    // Initialize stateLast as a copy of stateFirst
    DataGrid stateLast = {
        .colNum = stateFirst.colNum,
        .rowNum = stateFirst.rowNum,
        .U = calloc(stateFirst.colNum * stateFirst.rowNum, sizeof(int)),
        .T = calloc(stateFirst.colNum * stateFirst.rowNum, sizeof(double)),
        .f = calloc(stateFirst.colNum * stateFirst.rowNum, sizeof(uint8_t)),
        .set = NULL,
        .x = NULL,
        .y = NULL
    };

    memcpy(stateLast.f, stateFirst.f, stateFirst.colNum * stateFirst.rowNum * sizeof(uint8_t));
    memcpy(stateLast.T, stateFirst.T, stateFirst.colNum * stateFirst.rowNum * sizeof(double));
    memcpy(stateLast.U, stateFirst.U, stateFirst.colNum * stateFirst.rowNum * sizeof(int));

    // Thread creation and synchronization
    ThreadData data1 = {&stateFirst, true};
    ThreadData data2 = {&stateLast, false};

#if defined(_WIN32) || defined(_WIN64)
    // Windows threads
    HANDLE threads[2];
    threads[0] = CreateThread(NULL, 0, afmm_thread, &data1, 0, NULL);
    threads[1] = CreateThread(NULL, 0, afmm_thread, &data2, 0, NULL);
    WaitForMultipleObjects(2, threads, TRUE, INFINITE);
    CloseHandle(threads[0]);
    CloseHandle(threads[1]);
#else
    // POSIX threads
    pthread_t threads[2];
    pthread_create(&threads[0], NULL, afmm_thread, &data1);
    pthread_create(&threads[1], NULL, afmm_thread, &data2);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
#endif

    // Allocate output arrays
    *deltaU_out = calloc(img->width * img->height, sizeof(double));
    *DT_out = calloc(img->width * img->height, sizeof(double));

    // Process results
    for (int y = 1; y < stateFirst.rowNum - 1; y++) {
        for (int x = 1; x < stateFirst.colNum - 1; x++) {
            int oldIdx = y * stateFirst.colNum + x;
            if (mask[oldIdx] == 0) continue;
            
            int newIdx = (y-1) * img->width + (x - 1);
            (*DT_out)[newIdx] = stateFirst.T[oldIdx];
            
            // Calculate deltaU
            double deltaUFirst = 0;
            double deltaULast = 0;
            double difference;
            
            int neighbors[8];
            moore_neighborhood(oldIdx, stateFirst.colNum, neighbors);
            
            for (int i = 0; i < 8; i++) {
                int neighbor = neighbors[i];
                if (mask[neighbor] == 0) continue;
                
                // Calculate deltaUFirst
                if (stateFirst.set[stateFirst.U[neighbor]] != stateFirst.set[stateFirst.U[oldIdx]]) {
                    difference = INFINITY;
                } else {
                    difference = fabs((double)(stateFirst.U[neighbor] - stateFirst.U[oldIdx]));
                }
                
                if (deltaUFirst < difference) {
                    deltaUFirst = difference;
                }
                
                // Calculate deltaULast
                if (stateLast.set[stateLast.U[neighbor]] != stateLast.set[stateLast.U[oldIdx]]) {
                    difference = INFINITY;
                } else {
                    difference = fabs((double)(stateLast.U[neighbor] - stateLast.U[oldIdx]));
                }
                
                if (deltaULast < difference) {
                    deltaULast = difference;
                }
            }
            
            // Apply thresholds
            if (deltaUFirst < 3) deltaUFirst = 0;
            if (deltaULast < 3) deltaULast = 0;
            
            // Take minimum of both directions
            (*deltaU_out)[newIdx] = deltaUFirst;
            if (deltaULast < deltaUFirst) {
                (*deltaU_out)[newIdx] = deltaULast;
            }
        }
    }
    
    // Cleanup
    free(mask);
    free(stateFirst.U);
    free(stateFirst.T);
    free(stateFirst.f);
    free(stateFirst.set);
    free(stateFirst.x);
    free(stateFirst.y);
    free(stateLast.U);
    free(stateLast.T);
    free(stateLast.f);
    free(stateLast.set);
    free(stateLast.x);
    free(stateLast.y);
}

// Skeletonization function
void Skeletonize(const Image* img, double threshold, int rgb_binary, 
                 uint8_t** output_skeleton,
                 double** output_deltaU,
                 double** output_DT) {
    double* deltaU;
    double* DT;

    // Compute AFMM
    clock_t skel_start = clock();
    AFMM(img, rgb_binary, &deltaU, &DT);
    clock_t skel_end = clock();
    // print_elapsed_time(skel_start, skel_end, "AFMM computation");
    
    
    uint8_t* output = calloc(img->width * img->height, sizeof(uint8_t));
    
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int idx = x + y * img->width;
            if (deltaU[idx] > threshold) {
                output[idx] = 255;  // White pixel for skeleton
            } else {
                output[idx] = 0;    // Black pixel for background
            }
        }
    }
    
    // Set the output pointers
    *output_skeleton = output;
    *output_deltaU = deltaU;
    *output_DT = DT;
}

// Image loading function
Image* load_image(const char* filename, int is_rgb) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        fprintf(stderr, "Error loading image %s\n", filename);
        return NULL;
    }

    Image* img = malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->data = malloc(width * height * sizeof(uint32_t));

    if (!is_rgb) {
        // Convert to binary (0 or 1)
        for (int i = 0; i < width * height; i++) {
            if (channels == 1) {                   
                img->data[i] = (data[i] > 0) ? 1 : 0; 
            }
        }
    } else {
        // Convert to RGB (0-255)
        for (int i = 0; i < width * height; i++) {
            uint8_t r = (channels >= 1) ? data[i * channels] : 0;
            uint8_t g = (channels >= 2) ? data[i * channels + 1] : r;
            uint8_t b = (channels >= 3) ? data[i * channels + 2] : r;
            uint8_t a = (channels >= 4) ? data[i * channels + 3] : 0xFF;
            
            img->data[i] = (r << 24) | (g << 16) | (b << 8) | a;
        }
    }

    stbi_image_free(data);
    return img;
}

// Save distance transform to an image file
void save_distance_transform(const double* DT, int width, int height, const char* filename) {
    unsigned char* output = malloc(width * height);
    
    // Find maximum value for normalization
    double max_val = 0;
    for (int i = 0; i < width * height; i++) {
        if (DT[i] > max_val && !isinf(DT[i])) max_val = DT[i];
    }

    // Normalize and convert to 8-bit
    for (int i = 0; i < width * height; i++) {
        if (isinf(DT[i])) {
            output[i] = 255;
        } else {
            output[i] = (unsigned char)(255.0 * DT[i] / max_val);
        }
    }

    stbi_write_png(filename, width, height, 1, output, width);
    free(output);
}

// Save deltaU to an image file
void save_deltaU(const double* deltaU, int width, int height, const char* filename) {
    unsigned char* output = malloc(width * height);
    
    // Find maximum value for normalization
    double max_val = 0;
    for (int i = 0; i < width * height; i++) {
        if (deltaU[i] > max_val && !isinf(deltaU[i])) max_val = deltaU[i];
    }

    // Normalize and convert to 8-bit
    for (int i = 0; i < width * height; i++) {
        if (isinf(deltaU[i])) {
            output[i] = 255;
        } else {
            output[i] = (unsigned char)(255.0 * deltaU[i] / max_val);
        }
    }

    stbi_write_png(filename, width, height, 1, output, width);
    free(output);
}

// Save skeleton to an image file
void save_skeleton(const uint8_t* skeleton, int width, int height, const char* filename) {
    stbi_write_png(filename, width, height, 1, skeleton, width);
}

// Cleanup functions
void cleanup_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// Print elapsed time
void print_elapsed_time(clock_t start_time, clock_t end_time, const char* operation) {
    double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("%s took %.3f seconds\n", operation, elapsed);
}