/*
 * Copyright (C) 2008, 2009, 2010 Richard Membarth <richard.membarth@cs.fau.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with NVIDIA CUDA Software Development Kit (or a modified version of that
 * library), containing parts covered by the terms of a royalty-free,
 * non-exclusive license, the licensors of this Program grant you additional
 * permission to convey the resulting work.
 */

/* Implementation of multiresolution bilateral filter on the GPU using CUDA
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

// includes, kernels
#include "multi_res_device.cu"
#include "multi_res_host.hpp"
#include "gimp_main.hpp"
#include "gimp_gui.hpp"


// static variables
static float progress = 0.0f;
static float complexity = 0.0f;
static float num_col = 0;
static int initialized = 0;

////////////////////////////////////////////////////////////////////////////////
//Update gimp progress bar
////////////////////////////////////////////////////////////////////////////////
void update_progress_host(float factor) {
    num_col += factor;
    while (num_col >= complexity) {
        progress += 0.01f;
        num_col -= complexity;
        gimp_progress_update(progress);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Wrapper function to call decompose kernel(s) on the GPU
// use function overloading; function templates don't work anymore for CUDA2.0
////////////////////////////////////////////////////////////////////////////////
float decompose(int *g0, int *g1, int *l0, int data_width, int data_height) {
    unsigned int mem_size = data_width * data_height * sizeof(int);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x1 = 48, blocksize_y1 = 2;
    unsigned int blocksize_x2 = 32, blocksize_y2 = 2;

    if (data_width < 128) {
        blocksize_x1 = 16;
        blocksize_y1 = 16;
        blocksize_x2 = 16;
        blocksize_y2 = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // reduce
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_g0, g0, mem_size));
    lowpass_ds<<<dim3((int)ceil((float)(data_width/2)/blocksize_x1), (int)ceil((float)(data_height/2)/blocksize_y1), 1),
        dim3 (blocksize_x1, blocksize_y1), 0, 0>>>(g1, data_width, data_height);
    // expand + sub
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_g0, g0, mem_size));
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_g1, g1, mem_size/4));
    expand_sub<<<dim3((int)ceil((float)(data_width/2)/blocksize_x2), (int)ceil((float)(data_height/2)/blocksize_y2), 1),
        dim3 (blocksize_x2, blocksize_y2), 0, 0>>>(l0, data_width/2, data_height/2);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height);
#ifdef PRINT_TIMES
    fprintf(stderr, "decompose time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}
float decompose(float *g0, float *g1, float *l0, int data_width, int data_height) {
    unsigned int mem_size = data_width * data_height * sizeof(float);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x1 = 48, blocksize_y1 = 2;
    unsigned int blocksize_x2 = 32, blocksize_y2 = 2;

    if (data_width < 128) {
        blocksize_x1 = 16;
        blocksize_y1 = 16;
        blocksize_x2 = 16;
        blocksize_y2 = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // reduce
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_g0, g0, mem_size));
    lowpass_ds<<<dim3((int)ceil((float)(data_width/2)/blocksize_x1), (int)ceil((float)(data_height/2)/blocksize_y1), 1),
        dim3 (blocksize_x1, blocksize_y1), 0, 0>>>(g1, data_width, data_height);
    // expand + sub
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_g1, g1, mem_size/4));
    expand_sub<<<dim3((int)ceil((float)(data_width/2)/blocksize_x2), (int)ceil((float)(data_height/2)/blocksize_y2), 1),
        dim3 (blocksize_x2, blocksize_y2), 0, 0>>>(l0, data_width/2, data_height/2);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height);
#ifdef PRINT_TIMES
    fprintf(stderr, "decompose time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}


////////////////////////////////////////////////////////////////////////////////
//! Wrapper function to call filter kernel on the GPU
// use function overloading; function templates don't work anymore for CUDA2.0
////////////////////////////////////////////////////////////////////////////////
float filter(float *f0, float *l0, int data_width, int data_height, int sigma_d, int sigma_r) {
    unsigned int mem_size = data_width * data_height * sizeof(float);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x = 64, blocksize_y = 1;

    if (data_width < 64) {
        blocksize_x = 16;
        blocksize_y = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // filter
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_l0, l0, mem_size));
    bilateral_filter<<<dim3((int)ceil((float)data_width/blocksize_x), 
            (int)ceil((float)data_height/blocksize_y), 1), dim3 (blocksize_x, blocksize_y), 0, 0>>>
        (f0, data_width, data_height, sigma_d, sigma_r);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height*(2*sigma_d*2+1)*(2*sigma_d*2+1));
#ifdef PRINT_TIMES
    fprintf(stderr, "filter time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}
float filter(int *f0, int *l0, int data_width, int data_height, int sigma_d, int sigma_r) {
    unsigned int mem_size = data_width * data_height * sizeof(int);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x = 64, blocksize_y = 1;

    if (data_width < 64) {
        blocksize_x = 16;
        blocksize_y = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // filter
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_l0, l0, mem_size));
    bilateral_filter<<<dim3((int)ceil((float)data_width/blocksize_x), 
            (int)ceil((float)data_height/blocksize_y), 1), dim3 (blocksize_x, blocksize_y), 0, 0>>>
        (f0, data_width, data_height, sigma_d, sigma_r);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height*(2*sigma_d*2+1)*(2*sigma_d*2+1));
#ifdef PRINT_TIMES
    fprintf(stderr, "filter time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}


////////////////////////////////////////////////////////////////////////////////
//! Wrapper function to call reconstruct kernel on the GPU
// use function overloading; function templates don't work anymore for CUDA2.0
////////////////////////////////////////////////////////////////////////////////
float reconstruct(float *f0, float *r1, float *r0, int data_width, int data_height) {
    unsigned int mem_size = data_width * data_height * sizeof(float);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x = 32, blocksize_y = 2;

    if (data_width < 32) {
        blocksize_x = 16;
        blocksize_y = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // reconstruct
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_f0, f0, mem_size*4));
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_r1, r1, mem_size));
    reconstruct<<<dim3((int)ceil((float)data_width/blocksize_x), (int)ceil((float)data_height/blocksize_y), 1),
        dim3(blocksize_x, blocksize_y), 0, 0>>>(r0, data_width, data_height);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height);
#ifdef PRINT_TIMES
    fprintf(stderr, "reconstruct time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}
float reconstruct(int *f0, int *r1, int *r0, int data_width, int data_height) {
    unsigned int mem_size = data_width * data_height * sizeof(int);
    cudaEvent_t start, end;
    float kernel_time;
    // optimal configuration determined by configuration space exploration
    unsigned int blocksize_x = 32, blocksize_y = 2;

    if (data_width < 32) {
        blocksize_x = 16;
        blocksize_y = 16;
    }

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // reconstruct
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_f0, f0, mem_size*4));
    CUDA_SAFE_CALL(cudaBindTexture(0, tex_r1, r1, mem_size));
    reconstruct<<<dim3((int)ceil((float)data_width/blocksize_x), (int)ceil((float)data_height/blocksize_y), 1),
        dim3(blocksize_x, blocksize_y), 0, 0>>>(r0, data_width, data_height);

    CUDA_SAFE_CALL(cudaEventRecord(end, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, start, end));
    update_progress_host(data_height);
#ifdef PRINT_TIMES
    fprintf(stderr, "reconstruct time(%d:%d) GPU: %f (ms)\n", data_width, data_height, kernel_time);
    return kernel_time;
#else
    return 0.0f;
#endif
}


////////////////////////////////////////////////////////////////////////////////
//! Returns the number of CUDA capable GPUs
////////////////////////////////////////////////////////////////////////////////
int has_cuda_device(void) {
    int device_count;

    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));

    return device_count;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the multiresolution filter on the GPU
////////////////////////////////////////////////////////////////////////////////
template <typename T> void run_gpu_intern(guchar *host_g0, guchar *host_r0, const int width, const int height, const int channels, const int sigma_d, const int sigma_r) {
    unsigned int timer = 0, mem_size;
    size_t mem_free = 0, mem_total = 0;
    int data_width = 1, data_height = 1;
    int dev, tex_alignment;
    char *argv[2] = {"1337", "-quiet"};
    cudaDeviceProp device_prop;
    T *channel_g0 = NULL;
    T *g0 = NULL, *l0 = NULL;
    T *g1 = NULL, *l1 = NULL;
    T *g2 = NULL, *l2 = NULL;
    T *g3 = NULL, *l3 = NULL;
    T *g4 = NULL, *l4 = NULL;
    T *g5 = NULL, *l5 = NULL;
    float time;
    float gaussian[2*2*MAX_SIGMA_D+1];
    float gaussian_h[2*2*MAX_SIGMA_D+1][2*2*MAX_SIGMA_D+1];

    // initialize device and timer
    if (!initialized) {
        CUT_DEVICE_INIT(2, argv);
        initialized++;
    }
    // get texture alignment
    CUDA_SAFE_CALL(cudaGetDevice(&dev));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, dev));
    tex_alignment = device_prop.textureAlignment;


    // get next power of 2
    while (data_width < width) {
        data_width = data_width << 1;
    }
    while (data_height < height) {
        data_height = data_height << 1;
    }

    // calculate overall complexity: 31/16 decompose + 31/16 * filter_radius^2 + 31/16 reconstruct
    // norm to lines per 1percent
    complexity = ((31 + 31*(2*sigma_d*2+1)*(2*sigma_d*2+1) + 31)*channels*data_height)/(16*100);
    progress = 0.0f;
    num_col = 0;
    mem_size = sizeof(T) * data_width * data_height;
    // compute gaussian spread matrix for domain space
    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) gaussian[xf+2*sigma_d] = exp(-1/(2.0f*sigma_d*sigma_d)*(xf*xf));
    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
        for (int yf=-2*sigma_d; yf<=2*sigma_d; yf++) {
            gaussian_h[xf+2*sigma_d][yf+2*sigma_d] = gaussian[xf+2*sigma_d] * gaussian[yf+2*sigma_d];
        }
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gaussian_d, gaussian_h, (2*2*MAX_SIGMA_D+1)*(2*2*MAX_SIGMA_D+1)*sizeof(float)));

    // check if enough memory is available
    CUDA_SAFE_CALL(cudaMemGetInfo(&mem_free, &mem_total));
    if (mem_free < 4*mem_size) {
        g_message("Not enough memory on GPU:\n %d bytes free\n%d bytes required!", (int)mem_free, (int)mem_total);
        return;
    }
    // allocate && copy device memory
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    CUDA_SAFE_CALL(cudaMalloc((void**) &g0, mem_size * 2));
    channel_g0 = (T *) malloc(data_height*data_width*sizeof(T));
    g1 = &g0[(int)ceil((float)data_width*data_height*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    g2 = &g1[(int)ceil((float)data_width*(data_height/2)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    g3 = &g2[(int)ceil((float)data_width*(data_height/4)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    g4 = &g3[(int)ceil((float)data_width*(data_height/8)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    g5 = &g4[(int)ceil((float)data_width*(data_height/16)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    CUDA_SAFE_CALL(cudaMalloc((void**) &l0, mem_size * 2));
    l1 = &l0[(int)ceil((float)data_width*data_height*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    l2 = &l1[(int)ceil((float)data_width*(data_height/2)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    l3 = &l2[(int)ceil((float)data_width*(data_height/4)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    l4 = &l3[(int)ceil((float)data_width*(data_height/8)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
    l5 = &l4[(int)ceil((float)data_width*(data_height/16)*sizeof(T) / tex_alignment)*(tex_alignment/sizeof(T))];
#ifdef PRINT_TIMES
    fprintf(stderr, "\n#################################################################################\n");
    time = cutGetTimerValue(timer);
    fprintf(stderr, "Memory allocation time (GPU): %f (ms)\n", time);
#endif

    for (int i=0; i < channels; i++) {
        // pre-process data - copy guchars to new array, mirror pixel values at the margin (get power of 2 for width and height)
        CUT_SAFE_CALL(cutResetTimer(timer));
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                channel_g0[j*data_width + k] = host_g0[(width*j + k)*channels + i];
            }
            for (int k=width; k<data_width; k++) {
                channel_g0[j*data_width + k] = host_g0[(width*j + width-2-(k-width))*channels + i];
            }
        }
        for (int j=height; j<data_height; j++) {
            for (int k=0; k<data_width; k++) {
                channel_g0[j*data_width + k] = channel_g0[(height-2-(j-height))*data_width + k];
            }
        }
#ifdef PRINT_TIMES
        fprintf(stderr, "Data pre-processing time: %f (ms)\n", cutGetTimerValue(timer));
#endif

        // copy input data to device
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUDA_SAFE_CALL(cudaMemcpy(g0, channel_g0, data_width*data_height*sizeof(T), cudaMemcpyHostToDevice));
#ifdef PRINT_TIMES
        time += cutGetTimerValue(timer);
        fprintf(stderr, "Memory copy time host->device: %f (ms)\n", cutGetTimerValue(timer));
#endif


        // decompose image - the smallest resolution we get from gimp is 200x200 (256x256 padded)
        // for 256x256 we use one stage less - the lowest stage would have be 8x8
        time += decompose(g0, g1, l0, data_width, data_height);
        time += decompose(g1, g2, l1, data_width/2, data_height/2);
        time += decompose(g2, g3, l2, data_width/4, data_height/4);
        if (data_width > 256) {
            time += decompose(g3, g4, l3, data_width/8, data_height/8);
            time += decompose(g4, l5, l4, data_width/16, data_height/16);
        } else {
            time += decompose(g3, l4, l3, data_width/8, data_height/8);
        }

        // filter image - reuse g0 for f0
        time += filter(g0, l0, data_width, data_height, sigma_d, sigma_r);
        time += filter(g1, l1, data_width/2, data_height/2, sigma_d, sigma_r);
        time += filter(g2, l2, data_width/4, data_height/4, sigma_d, sigma_r);
        time += filter(g3, l3, data_width/8, data_height/8, sigma_d, sigma_r);
        time += filter(g4, l4, data_width/16, data_height/16, sigma_d, sigma_r);
        if (data_width > 256) {
            time += filter(g5, l5, data_width/32, data_height/32, sigma_d, sigma_r);
        }

        // reconstruct image - reuse l0 for r0
        if (data_width > 256) {
            time += reconstruct(g4, g5, l4, data_width/32, data_height/32);
            time += reconstruct(g3, l4, l3, data_width/16, data_height/16);
        } else {
            time += reconstruct(g3, g4, l3, data_width/16, data_height/16);
        }
        time += reconstruct(g2, l3, l2, data_width/8, data_height/8);
        time += reconstruct(g1, l2, l1, data_width/4, data_height/4);
        time += reconstruct(g0, l1, l0, data_width/2, data_height/2);

        // get result image
        CUT_SAFE_CALL(cutResetTimer(timer));
        CUDA_SAFE_CALL(cudaMemcpy(channel_g0, l0, mem_size, cudaMemcpyDeviceToHost));
#ifdef PRINT_TIMES
        time += cutGetTimerValue(timer);
        fprintf(stderr, "Memory copy time device->host: %f (ms)\n", cutGetTimerValue(timer));
#endif

        // post process image - we don't need the pixels added by the padding
        CUT_SAFE_CALL(cutResetTimer(timer));
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                host_r0[(width*j + k)*channels + i] = (channel_g0[j*data_width + k] < 0)?0:
                    (channel_g0[j*data_width + k]>255)?255:(guchar)channel_g0[j*data_width + k];
            }
        }
#ifdef PRINT_TIMES
        time += cutGetTimerValue(timer);
        fprintf(stderr, "Data post-processing time: %f (ms)\n", cutGetTimerValue(timer));
#endif
    }
#ifdef PRINT_TIMES
        fprintf(stderr, "Total time: %f (ms)\n", time);
        fprintf(stderr, "#################################################################################\n\n");
#endif

    // cleanup memory
    CUT_SAFE_CALL(cutDeleteTimer(timer));
    CUDA_SAFE_CALL(cudaFree(g0));
    CUDA_SAFE_CALL(cudaFree(l0));
}


////////////////////////////////////////////////////////////////////////////////
//! Run the multiresolution filter on the GPU
////////////////////////////////////////////////////////////////////////////////
void run_gpu(guchar *host_g0, guchar *host_r0, const int width, const int height, const int channels, const int sigma_d, const int sigma_r, const int use_float) {
    if (use_float) run_gpu_intern<float>(host_g0, host_r0, width, height, channels, sigma_d, sigma_r);
    else run_gpu_intern<int32_t>(host_g0, host_r0, width, height, channels, sigma_d, sigma_r);
}

