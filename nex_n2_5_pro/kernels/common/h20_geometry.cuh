// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Launch geometry of the kernels that have an H20 variant. Geometry 0 is the released grid.
// Geometry 1 gives each thread block more rows, so that the grid fits one round of thread blocks
// (fewer rounds for the CP front) on a GPU with fewer SMs (H20: 78), where the released grid needs
// a partial extra round. Both geometries compute each output with the same operations in the same
// order, so they give bitwise-identical outputs.
//
// NEXRT_H20_GEOMETRY=0 or =1 forces the geometry. Unset or empty selects geometry 1 when the
// device has fewer than 128 SMs, else geometry 0 (H100 and H200: 132 SMs). Any other value is an
// error.
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
namespace {  // internal linkage: each kernel library keeps its own cache

// Returns 0 or 1, or a negative CUDA error code.
inline int query_h20_geometry(int device) {
    const char* env = std::getenv("NEXRT_H20_GEOMETRY");
    if (env != nullptr && env[0] != '\0') {
        if (std::strcmp(env, "0") == 0) return 0;
        if (std::strcmp(env, "1") == 0) return 1;
        return -static_cast<int>(cudaErrorInvalidValue);
    }
    int sms = 0;
    const cudaError_t err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) return -static_cast<int>(err);
    return sms < 128 ? 1 : 0;
}

// The geometry of the current device: 0 or 1, or a negative CUDA error code. The query runs once
// per process and device, and later launches read the cached value.
inline int h20_geometry() {
    constexpr int kCachedDevices = 64;
    static std::atomic<int> cached[kCachedDevices];  // 0 until the first query, then geometry + 1
    int device = 0;
    const cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -static_cast<int>(err);
    if (device < 0 || device >= kCachedDevices) return query_h20_geometry(device);
    int entry = cached[device].load(std::memory_order_relaxed);
    if (entry == 0) {
        const int geometry = query_h20_geometry(device);
        if (geometry < 0) return geometry;
        entry = geometry + 1;
        cached[device].store(entry, std::memory_order_relaxed);
    }
    return entry - 1;
}

}  // namespace
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
