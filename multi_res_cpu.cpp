/*
 * Copyright (C) 2008, 2009, 2010, 2012 Richard Membarth <richard.membarth@cs.fau.de>
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
 */

#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gimp_main.hpp"
#include "gimp_gui.hpp"
#include "defines_cpu.hpp"


// forward declarations
template <typename T> void lowpass(T *reference, T *idata, const int width, const int height);
template <typename T> void downsample(T *reference, T *idata, const int width, const int height);
template <typename T> void sub(T *reference, T *g0, T *g1e, const int width, const int height);
template <typename T> void add(T *reference, T *f0, T *r1e, const int width, const int height);
template <typename T> void expand(T *reference, T *g0, const int width, const int height);
template <typename T> void bilateral_filter(T *reference, T *l0, const int width, const int height, int sigma_d, int sigma_r);

template <typename T> void reduce(T *g1, T *g0, const int width, const int height);
template <typename T> float decompose(T *l0, T *g1, T *g0, const int width, const int height);
template <typename T> float filter(T *f0, T *l0, const int width, const int height, int sigma_d, int sigma_r);
template <typename T> float reconstruct(T *r0, T *r1, T *f0, const int width, const int height);

// static variables
static float progress = 0;
static float complexity = 0;
static float num_col = 0;

void update_progress(float factor) {
    num_col += factor;
    while (num_col >= complexity) {
        progress += 0.01f;
        num_col -= complexity;
        gimp_progress_update(progress);
    }
}


// lowpass operation
template <typename T> void lowpass(T *reference, T *idata, const int width, const int height) {
    for (int y=0; y<height; y++) {
        int top = 1;
        int bottom = 1;
        if (y==0) top = -1;
        if (y==(height-1)) bottom = -1;
        for (int x=0; x<width; x++) {
            int left = 1;
            int right = 1;
            if (x==0) left = -1;
            if (x==(width-1)) right = -1;
            reference[y*width + x] =
                idata[(y-top)*width + x-left]/16 +
                idata[(y-top)*width + x]/8 +
                idata[(y-top)*width + x+right]/16 +
                idata[y*width + x-left]/8 +
                idata[y*width + x]/4 +
                idata[y*width + x+right]/8 +
                idata[(y+bottom)*width + x-left]/16 +
                idata[(y+bottom)*width + x]/8 +
                idata[(y+bottom)*width + x+right]/16;
        }
        update_progress(1);
    }
}

// downsample operation
template <typename T> void downsample(T *reference, T *idata, const int width, const int height) {
    for (int y=0; y<height; y=y+2) {
        for (int x=0; x<width; x=x+2) {
            reference[(y/2)*width/2 + x/2] = idata[y*width + x];
        }
        update_progress(1);
    }
}

// sub operation
template <typename T> void sub(T *reference, T *g0, T *g1e, const int width, const int height) {
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            reference[y*width + x] = g0[y*width + x] - g1e[y*width + x];
        }
        update_progress(1);
    }
}

// add operation
template <typename T> void add(T *reference, T *f0, T *r1e, const int width, const int height) {
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            reference[y*width + x] = f0[y*width + x] + r1e[y*width + x];
        }
        update_progress(1);
    }
}

// expand operation
template <typename T> void expand(T *reference, T *g0, const int width, const int height) {
    for (int y=0; y<height; y++) {
        int bottom = (y == height-1) ? 0:width;
        for (int x=0; x<width; x++) {
            int right = (x == width-1) ? 0:1;
            reference[2*x   + 4*y*width] = g0[y*width + x];
            reference[2*x+1 + 4*y*width] = g0[y*width + x]/2 + g0[x+right + y*width]/2;
            reference[2*x   + 4*y*width + 2*width] = g0[y*width + x]/2 + g0[x + y*width+bottom]/2;
            reference[2*x+1 + 4*y*width + 2*width] = g0[y*width + x]/4 + g0[x+right + y*width]/4 + g0[x + y*width+bottom]/4 + g0[x+right + y*width+bottom]/4;
        }
        update_progress(1);
    }
}

// bilateral filter operation
template <typename T> void bilateral_filter(T *reference, T *l0, const int width, const int height, int sigma_d, int sigma_r) {
    int fwin = 2*2*sigma_d+1;
    gfloat c_r = 1.0f/(2.0f*sigma_r*sigma_r);
    gfloat c_d = 1.0f/(2.0f*sigma_d*sigma_d);
    gfloat gaussian[fwin];
    gfloat gaussian_d[fwin][fwin];   // gaussian_d is the geometric spread matrix.
    gfloat s, d, p;
    T diff;

    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) gaussian[xf+2*sigma_d] = expf(-c_d*(xf*xf));
    for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
        for (int yf=-2*sigma_d; yf<=2*sigma_d; yf++) {
            gaussian_d[xf+2*sigma_d][yf+2*sigma_d] = gaussian[xf+2*sigma_d] * gaussian[yf+2*sigma_d];
        }
    }

    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            s = 0;
            d = 0;
            p = 0;
            for (int yf=-2*sigma_d; yf<=2*sigma_d; yf++) {
                int y_corr = y + yf;
                #if 0
                y_corr = (y_corr < 0) ? -y_corr : y_corr;
                y_corr = (y+yf >= height) ? height-1 - (1+y_corr-height) : y_corr;
                #else
                if (y_corr < 0) y_corr = 0;
                if (y_corr >= height) y_corr = height - 1;
                #endif
                for (int xf=-2*sigma_d; xf<=2*sigma_d; xf++) {
                    int x_corr = x + xf;
                    #if 0
                    x_corr = (x_corr < 0) ? -x_corr : x_corr;
                    x_corr = (x+xf >= width) ? width-1 - (1+x_corr-width) : x_corr;
                    #else
                    if (x_corr < 0) x_corr = 0;
                    if (x_corr >= width) x_corr = width - 1;
                    #endif
                    diff = l0[y_corr*width + x_corr] - l0[y*width + x];
                    s = expf(-c_r * diff*diff) * gaussian_d[xf+2*sigma_d][yf+2*sigma_d];
                    d += s;
                    p += s * l0[y_corr*width + x_corr];
                }
            }
            reference[y*width + x] = (T) (p / d);
        }
        update_progress((2*sigma_d*2+1)*(2*sigma_d*2+1));
    }
}


// reduce operation
template <typename T> void reduce(T *g1, T *g0, const int width, const int height) {
    T *tmp = g_new(T, width*height);

    lowpass(tmp, g0, width, height);
    downsample(g1, tmp, width, height);

    g_free(tmp);
}


// decompose operation
template <typename T> float decompose(T *l0, T *g1, T *g0, const int width, const int height) {
    #ifdef PRINT_TIMES
    double time, start_time, end_time;

    start_time = get_time_ms();
    #endif
    reduce(g1, g0, width, height);
    expand(l0, g1, width/2, height/2);
    sub(l0, g0, l0, width, height);
    #ifdef PRINT_TIMES
    end_time = get_time_ms();
    time = end_time - start_time;
    fprintf(stderr, "decompose time (%dx%d) CPU: %f (ms)\n", width, height, time);
    return time;
    #else
    return 0;
    #endif
}


// filter operation
template <typename T> float filter(T *f0, T *l0, const int width, const int height, int sigma_d, int sigma_r) {
    #ifdef PRINT_TIMES
    double time, start_time, end_time;

    start_time = get_time_ms();
    #endif
    bilateral_filter(f0, l0, width, height, sigma_d, sigma_r);
    #ifdef PRINT_TIMES
    end_time = get_time_ms();
    time = end_time - start_time;
    fprintf(stderr, "filter time (%dx%d) CPU: %f (ms)\n", width, height, time);
    return time;
    #else
    return 0;
    #endif
}


// reconstruct operation
template <typename T> float reconstruct(T *r0, T *r1, T *f0, const int width, const int height) {
    #ifdef PRINT_TIMES
    double time, start_time, end_time;

    start_time = get_time_ms();
    #endif
    expand(r0, r1, width, height);
    add(r0, f0, r0, width*2, height*2);
    #ifdef PRINT_TIMES
    end_time = get_time_ms();
    time = end_time - start_time;
    fprintf(stderr, "reconstruct time (%dx%d) CPU: %f (ms)\n", width, height, time);
    return time;
    #else
    return 0;
    #endif
}


////////////////////////////////////////////////////////////////////////////////
//! Run the multiresolution filter on the CPU
////////////////////////////////////////////////////////////////////////////////
template <typename T> void run_cpu_intern(guchar *host_g0, guchar *host_r0, int width, int height, int channels, int sigma_d, int sigma_r) {
    unsigned int mem_size;
    int data_width = 1, data_height = 1;
    T *g0 = NULL, *l0 = NULL;
    T *g1 = NULL, *l1 = NULL;
    T *g2 = NULL, *l2 = NULL;
    T *g3 = NULL, *l3 = NULL;
    T *g4 = NULL, *l4 = NULL;
    T *g5 = NULL, *l5 = NULL;
    #ifdef PRINT_TIMES
    double time, start_time, end_time;
    #endif
    double total_time;

    // get next power of 2 -> padding of image
    while (data_width < width) {
        data_width = data_width << 1;
    }
    while (data_height < height) {
        data_height = data_height << 1;
    }
    // calculate overall complexity: 31/16 decompose + 31/16 * filter_radius^2 + 31/16 reconstruct
    // norm to lines per 1percent
    complexity = ((31 + 31*(2*sigma_d*2+1)*(2*sigma_d*2+1) + 31)*channels*data_height)/(16*100);
    progress = 0;
    num_col = 0;
    mem_size = sizeof(T) * data_width * data_height;

    #ifdef PRINT_TIMES
    start_time = get_time_ms();
    #endif
    // allocate && copy device memory
    g0 = g_new(T, mem_size*2);
    g1 = &g0[data_width*data_height];
    g2 = &g1[data_width*data_height/2];
    g3 = &g2[data_width*data_height/4];
    g4 = &g3[data_width*data_height/8];
    g5 = &g4[data_width*data_height/16];
    l0 = g_new(T, mem_size*2);
    l1 = &l0[data_width*data_height];
    l2 = &l1[data_width*(data_height/2)];
    l3 = &l2[data_width*(data_height/4)];
    l4 = &l3[data_width*(data_height/8)];
    l5 = &l4[data_width*(data_height/16)];
    #ifdef PRINT_TIMES
    fprintf(stderr, "\n#################################################################################\n");
    end_time = get_time_ms();
    time = end_time - start_time;
    total_time = time;
    fprintf(stderr, "Memory allocation time (CPU): %f (ms)\n", time);
    #endif

    for (int i=0; i < channels; i++) {
        // pre-process data - copy guchars to new array, mirror pixel values at the margin (get power of 2 for width and height)
        #ifdef PRINT_TIMES
        start_time = get_time_ms();
        #endif
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                g0[j*data_width + k] = host_g0[(width*j + k)*channels + i];
            }
            for (int k=width; k<data_width; k++) {
                g0[j*data_width + k] = host_g0[(width*j + width-2-(k-width))*channels + i];
            }
        }
        for (int j=height; j<data_height; j++) {
            for (int k=0; k<data_width; k++) {
                g0[j*data_width + k] = g0[(height-2-(j-height))*data_width + k];
            }
        }
        #ifdef PRINT_TIMES
        end_time = get_time_ms();
        time = end_time - start_time;
        total_time += time;
        fprintf(stderr, "Data pre-processing time: %f (ms)\n", time);
        #endif

        // decompose image
        total_time += decompose(l0, g1, g0, data_width, data_height);
        total_time += decompose(l1, g2, g1, data_width/2, data_height/2);
        total_time += decompose(l2, g3, g2, data_width/4, data_height/4);
        total_time += decompose(l3, g4, g3, data_width/8, data_height/8);
        total_time += decompose(l4, l5, g4, data_width/16, data_height/16);

        // filter image - reuse g0 for f0
        total_time += filter(g0, l0, data_width, data_height, sigma_d, sigma_r);
        total_time += filter(g1, l1, data_width/2, data_height/2, sigma_d, sigma_r);
        total_time += filter(g2, l2, data_width/4, data_height/4, sigma_d, sigma_r);
        total_time += filter(g3, l3, data_width/8, data_height/8, sigma_d, sigma_r);
        total_time += filter(g4, l4, data_width/16, data_height/16, sigma_d, sigma_r);
        total_time += filter(g5, l5, data_width/32, data_height/32, sigma_d, sigma_r);

        // reconstruct image - reuse l0 for r0
        total_time += reconstruct(l4, g5, g4, data_width/32, data_height/32);
        total_time += reconstruct(l3, l4, g3, data_width/16, data_height/16);
        total_time += reconstruct(l2, l3, g2, data_width/8, data_height/8);
        total_time += reconstruct(l1, l2, g1, data_width/4, data_height/4);
        total_time += reconstruct(l0, l1, g0, data_width/2, data_height/2);

        // post process image - we don't need the pixels added by the padding
        #ifdef PRINT_TIMES
        start_time = get_time_ms();
        #endif
        for (int j=0; j<height; j++) {
            for (int k=0; k<width; k++) {
                host_r0[(width*j + k)*channels + i] = (guchar)((l0[j*data_width + k] < 0)?0:(l0[j*data_width + k]>255)?255:l0[j*data_width + k]);
            }
        }
        #ifdef PRINT_TIMES
        end_time = get_time_ms();
        time = end_time - start_time;
        total_time += time;
        fprintf(stderr, "Data post-processing time: %f (ms)\n", time);
        #endif
    }
    #ifdef PRINT_TIMES
    fprintf(stderr, "Total time: %f (ms)\n", total_time);
    fprintf(stderr, "#################################################################################\n\n");
    #endif

    // cleanup memory
    g_free(g0);
    g_free(l0);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the multiresolution filter on the CPU
////////////////////////////////////////////////////////////////////////////////
void run_cpu(guchar *host_g0, guchar *host_r0, int width, int height, int channels, int sigma_d, int sigma_r, int use_float) {
    if (use_float) {
        run_cpu_intern<gfloat>(host_g0, host_r0, width, height, channels, sigma_d, sigma_r);
    } else {
        run_cpu_intern<gint16>(host_g0, host_r0, width, height, channels, sigma_d, sigma_r);
    }
}

