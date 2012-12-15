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

#ifndef __DEFINES_GPU_HPP__
#define __DEFINES_GPU_HPP__

#include "defines_cpu.hpp"

__device__ __constant__ float gaussian_d[2*2*MAX_SIGMA_D+1][2*2*MAX_SIGMA_D+1];

#endif /* __DEFINES_GPU_HPP__ */

