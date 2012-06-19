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

/* Implementation of lowpass filter on the GPU using CUDA
 * using global memory or texture memory
 * Device code.
 */

#ifndef __MULTI_RES_DEVICE_CU__
#define __MULTI_RES_DEVICE_CU__

#include <cutil.h>
#include <stdio.h>
#include "defines_gpu.hpp"


//Input data texture reference
#define get_data(tex_, x, y) (tex1Dfetch(tex_, ((x)+(y)*data_width)))
texture<int32_t, 1, cudaReadModeElementType> tex_g0;
texture<int32_t, 1, cudaReadModeElementType> tex_g1;
texture<int32_t, 1, cudaReadModeElementType> tex_g1e;
texture<int32_t, 1, cudaReadModeElementType> tex_l0;
texture<int32_t, 1, cudaReadModeElementType> tex_f0;
texture<int32_t, 1, cudaReadModeElementType> tex_r1;
texture<int32_t, 1, cudaReadModeElementType> tex_r1e;
texture<int32_t, 1, cudaReadModeElementType> tex_tmp;


// lowpass_ds: lowpass and downsample == reduce
__global__ void lowpass_ds(float *dev_odata, int data_width, int data_height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int top = 1, bottom = 1, left = 1, right = 1;

    if (2*ix < data_width && 2*iy < data_height) {
        if (iy == 0) top = -1;
        if (blockIdx.x+threadIdx.x == 0) left = -1;
        dev_odata[ix + iy*data_width/2] =
            __int_as_float(get_data(tex_g0, 2*ix-left,     2*iy-top))/16 +
            __int_as_float(get_data(tex_g0, 2*ix,          2*iy-top))/8 +
            __int_as_float(get_data(tex_g0, 2*ix+right,    2*iy-top))/16 +
            __int_as_float(get_data(tex_g0, 2*ix-left,     2*iy))/8 +
            __int_as_float(get_data(tex_g0, 2*ix,          2*iy))/4 +
            __int_as_float(get_data(tex_g0, 2*ix+right,    2*iy))/8 +
            __int_as_float(get_data(tex_g0, 2*ix-left,     2*iy+bottom))/16 +
            __int_as_float(get_data(tex_g0, 2*ix,          2*iy+bottom))/8 +
            __int_as_float(get_data(tex_g0, 2*ix+right,    2*iy+bottom))/16;
    }
}
__global__ void lowpass_ds(int *dev_odata, int data_width, int data_height) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int top = 1, bottom = 1, left = 1, right = 1;

    if (2*ix < data_width && 2*iy < data_height) {
        if (iy == 0) top = -1;
        if (blockIdx.x+threadIdx.x == 0) left = -1;
        dev_odata[ix + iy*data_width/2] =
            get_data(tex_g0, 2*ix-left,     2*iy-top)/16 +
            get_data(tex_g0, 2*ix,          2*iy-top)/8 +
            get_data(tex_g0, 2*ix+right,    2*iy-top)/16 +
            get_data(tex_g0, 2*ix-left,     2*iy)/8 +
            get_data(tex_g0, 2*ix,          2*iy)/4 +
            get_data(tex_g0, 2*ix+right,    2*iy)/8 +
            get_data(tex_g0, 2*ix-left,     2*iy+bottom)/16 +
            get_data(tex_g0, 2*ix,          2*iy+bottom)/8 +
            get_data(tex_g0, 2*ix+right,    2*iy+bottom)/16;
    }
}


// expand_sub: upsample and lowpass; subtact result from g0
__global__ void expand_sub (float *dev_odata, int data_width, int data_height) {
    // shared memory for reduce result - maximal configuration has 256 threads
    __shared__ float data[4*256];
    int smem_pos = 2*threadIdx.y*2*blockDim.x + 2*threadIdx.x;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int bottom = 1, right = 1;
    float reg = 0, regx = 0, regy = 0, regxy = 0;

    if (ix < data_width && iy < data_height) {
        if (iy == data_height-1) bottom = 0;
        if (ix == data_width-1) right = 0;
        reg   = __int_as_float(get_data(tex_g1, ix,         iy));
        regy  = __int_as_float(get_data(tex_g1, ix,         iy+bottom));
        regx  = __int_as_float(get_data(tex_g1, ix+right,   iy));
        regxy = __int_as_float(get_data(tex_g1, ix+right,   iy+bottom));
        // store data to shared memory, otherwise we have un-coalesced memory writes
        data[smem_pos] = __int_as_float(get_data(tex_g0, 2*ix, 4*iy)) - reg;
        data[smem_pos+1] = __int_as_float(get_data(tex_g0, 2*ix+1, 4*iy)) - (reg/2 + regx/2);
        data[smem_pos + 2*blockDim.x] = __int_as_float(get_data(tex_g0, 2*ix, 4*iy + 2)) - (reg/2 + regy/2);
        data[smem_pos+1 + 2*blockDim.x] = __int_as_float(get_data(tex_g0, 2*ix+1, 4*iy + 2)) - (reg/4 + regx/4 + regy/4 + regxy/4);
    }

    __syncthreads();

    if (ix < data_width && iy < data_height) {
        // write data coalesced to global memory
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width] = data[smem_pos - threadIdx.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + blockDim.x] = data[smem_pos -threadIdx.x + blockDim.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + 2*data_width] = data[smem_pos - threadIdx.x + 2*blockDim.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + 2*data_width + blockDim.x] = data[smem_pos - threadIdx.x + 2*blockDim.x + blockDim.x];
    }
}
__global__ void expand_sub (int *dev_odata, int data_width, int data_height) {
    // shared memory for reduce result - maximal configuration has 256 threads
    __shared__ int data[4*256];
    int smem_pos = 2*threadIdx.y*2*blockDim.x + 2*threadIdx.x;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int bottom = 1, right = 1;
    int reg = 0, regx = 0, regy = 0, regxy = 0;

    if (ix < data_width && iy < data_height) {
        if (iy == data_height-1) bottom = 0;
        if (ix == data_width-1) right = 0;
        reg   = get_data(tex_g1, ix,         iy);
        regy  = get_data(tex_g1, ix,         iy+bottom);
        regx  = get_data(tex_g1, ix+right,   iy);
        regxy = get_data(tex_g1, ix+right,   iy+bottom);
        // store data to shared memory, otherwise we have un-coalesced memory writes
        data[smem_pos] = get_data(tex_g0, 2*ix, 4*iy) - reg;
        data[smem_pos+1] = get_data(tex_g0, 2*ix+1, 4*iy) - (reg/2 + regx/2);
        data[smem_pos + 2*blockDim.x] = get_data(tex_g0, 2*ix, 4*iy + 2) - (reg/2 + regy/2);
        data[smem_pos+1 + 2*blockDim.x] = get_data(tex_g0, 2*ix+1, 4*iy + 2) - (reg/4 + regx/4 + regy/4 + regxy/4);
    }

    __syncthreads();

    if (ix < data_width && iy < data_height) {
        // write data coalesced to global memory
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width] = data[smem_pos - threadIdx.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + blockDim.x] = data[smem_pos -threadIdx.x + blockDim.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + 2*data_width] = data[smem_pos - threadIdx.x + 2*blockDim.x];
        dev_odata[2*ix - threadIdx.x    + 4*iy*data_width + 2*data_width + blockDim.x] = data[smem_pos - threadIdx.x + 2*blockDim.x + blockDim.x];
    }
}


__global__ void reconstruct(float *r0, int data_width, int data_height) {
    // shared memory for reduce result - maximal configuration has 256 threads
    __shared__ float data[4*256];
    int smem_pos = 2*threadIdx.y*2*blockDim.x + 2*threadIdx.x;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int bottom = 1, right = 1;
    float reg = 0, regx = 0, regy = 0, regxy = 0;

    if (ix < data_width && iy < data_height) {
        if (iy == data_height-1) bottom = 0;
        if (ix == data_width-1) right = 0;
        reg   = __int_as_float(get_data(tex_r1, ix,        iy));
        regy  = __int_as_float(get_data(tex_r1, ix,        iy+bottom));
        regx  = __int_as_float(get_data(tex_r1, ix+right,  iy));
        regxy = __int_as_float(get_data(tex_r1, ix+right,  iy+bottom));
        // data_width refers now to f0
        data_width *= 2;
        // store data to shared memory, otherwise we have un-coalesced memory writes
        data[smem_pos] = reg + __int_as_float(get_data(tex_f0, 2*ix, 2*iy));
        data[smem_pos+1] = reg/2 + regx/2 + __int_as_float(get_data(tex_f0, 2*ix+1, 2*iy));
        data[smem_pos + 2*blockDim.x] = reg/2 + regy/2 + __int_as_float(get_data(tex_f0, 2*ix, 2*iy+1));
        data[smem_pos+1 + 2*blockDim.x] = reg/4 + regx/4 + regy/4 + regxy/4 + __int_as_float(get_data(tex_f0, 2*ix+1, 2*iy+1));
    }

   __syncthreads();

    if (ix < data_width && iy < data_height) {
        // write data coalesced to global memory
        r0[2*ix - threadIdx.x    + 2*iy*data_width] = data[smem_pos - threadIdx.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + blockDim.x] = data[smem_pos -threadIdx.x + blockDim.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + data_width] = data[smem_pos - threadIdx.x + 2*blockDim.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + data_width + blockDim.x] = data[smem_pos - threadIdx.x + 2*blockDim.x + blockDim.x];
    }
}
__global__ void reconstruct(int *r0, int data_width, int data_height) {
    // shared memory for reduce result - maximal configuration has 256 threads
    __shared__ int data[4*256];
    int smem_pos = 2*threadIdx.y*2*blockDim.x + 2*threadIdx.x;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int bottom = 1, right = 1;
    int reg = 0, regx = 0, regy = 0, regxy = 0;

    if (ix < data_width && iy < data_height) {
        if (iy == data_height-1) bottom = 0;
        if (ix == data_width-1) right = 0;
        reg   = get_data(tex_r1, ix,        iy);
        regy  = get_data(tex_r1, ix,        iy+bottom);
        regx  = get_data(tex_r1, ix+right,  iy);
        regxy = get_data(tex_r1, ix+right,  iy+bottom);
        // data_width refers now to f0
        data_width *= 2;
        // store data to shared memory, otherwise we have un-coalesced memory writes
        data[smem_pos] = reg + get_data(tex_f0, 2*ix, 2*iy);
        data[smem_pos+1] = reg/2 + regx/2 + get_data(tex_f0, 2*ix+1, 2*iy);
        data[smem_pos + 2*blockDim.x] = reg/2 + regy/2 + get_data(tex_f0, 2*ix, 2*iy+1);
        data[smem_pos+1 + 2*blockDim.x] = reg/4 + regx/4 + regy/4 + regxy/4 + get_data(tex_f0, 2*ix+1, 2*iy+1);
    }

   __syncthreads();

    if (ix < data_width && iy < data_height) {
        // write data coalesced to global memory
        r0[2*ix - threadIdx.x    + 2*iy*data_width] = data[smem_pos - threadIdx.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + blockDim.x] = data[smem_pos -threadIdx.x + blockDim.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + data_width] = data[smem_pos - threadIdx.x + 2*blockDim.x];
        r0[2*ix - threadIdx.x    + 2*iy*data_width + data_width + blockDim.x] = data[smem_pos - threadIdx.x + 2*blockDim.x + blockDim.x];
    }
}


// bilater filter implementation; use lookup-table for gaussian spread calculations
__global__ void bilateral_filter(float *dev_odata, int data_width, int data_height, int sigma_d, int sigma_r) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int x_corr, y_corr;
    float c_r = 1.0f/(2.0f*sigma_r*sigma_r); 
    float s = 0.0f, d = 0.0f, p = 0.0f;
    float diff;

    if (ix < data_width && iy < data_height) {
        for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
            y_corr = iy + yf;
            #if 0
            y_corr = (y_corr < 0) ? -y_corr : y_corr;
            y_corr = (iy+yf >= data_height) ? data_height-1 - (1+y_corr-data_height) : y_corr;
            #else
            if (y_corr < 0) y_corr = 0;
            if (y_corr >= data_height) y_corr = data_height - 1;
            #endif
            for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                x_corr = ix + xf;
                #if 0
                x_corr = (x_corr < 0) ? -x_corr : x_corr;
                x_corr = (ix+xf >= data_width) ? data_width-1 - (1+x_corr-data_width) : x_corr;
                #else
                if (x_corr < 0) x_corr = 0;
                if (x_corr >= data_width) x_corr = data_width - 1;
                #endif
                diff = __int_as_float(get_data(tex_l0, x_corr, y_corr)) - __int_as_float(get_data(tex_l0, ix, iy));
                s = expf(-c_r * diff*diff) * gaussian_d[xf+2*sigma_d][yf+2*sigma_d];
                d += s;
                p += s * __int_as_float(get_data(tex_l0, x_corr, y_corr));
            }
        }
        dev_odata[ix + iy*data_width] = (float) (p / d);
    }
}
__global__ void bilateral_filter(int *dev_odata, int data_width, int data_height, int sigma_d, int sigma_r) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int x_corr, y_corr;
    float c_r = 1.0f/(2.0f*sigma_r*sigma_r); 
    float s = 0.0f, d = 0.0f, p = 0.0f;
    float diff;

    if (ix < data_width && iy < data_height) {
        for (int yf = -2*sigma_d; yf<=2*sigma_d; yf++) {
            y_corr = iy + yf;
            #if 0
            y_corr = (y_corr < 0) ? -y_corr : y_corr;
            y_corr = (iy+yf >= data_height) ? data_height-1 - (1+y_corr-data_height) : y_corr;
            #else
            if (y_corr < 0) y_corr = 0;
            if (y_corr >= data_height) y_corr = data_height - 1;
            #endif
            for (int xf = -2*sigma_d; xf<=2*sigma_d; xf++) {
                x_corr = ix + xf;
                #if 0
                x_corr = (x_corr < 0) ? -x_corr : x_corr;
                x_corr = (ix+xf >= data_width) ? data_width-1 - (1+x_corr-data_width) : x_corr;
                #else
                if (x_corr < 0) x_corr = 0;
                if (x_corr >= data_width) x_corr = data_width - 1;
                #endif
                diff = get_data(tex_l0, x_corr, y_corr) - get_data(tex_l0, ix, iy);
                s = expf(-c_r * diff*diff) * gaussian_d[xf+2*sigma_d][yf+2*sigma_d];
                d += s;
                p += s * get_data(tex_l0, x_corr, y_corr);
            }
        }
        dev_odata[ix + iy*data_width] = (int) (p / d);
    }
}

#endif // #ifndef __MULTI_RES_DEVICE_CU__

