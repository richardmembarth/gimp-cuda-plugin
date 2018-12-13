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
 *
 * Additional permission under GNU GPL version 3 section 7
 */

#ifndef __GIMP_MAIN_HPP__
#define __GIMP_MAIN_HPP__

#include <sys/time.h>

typedef struct {
  gint sigma_d;
  gint sigma_r;
  gboolean preview;
  gboolean gpu;
  gboolean use_float;
} filter_config;

extern filter_config filter_vals;

extern void run_multi_filter(GimpDrawable *drawable, GimpPreview *preview);

// get time in milliseconds
inline double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}

#endif /* __GIMP_MAIN_HPP__ */

