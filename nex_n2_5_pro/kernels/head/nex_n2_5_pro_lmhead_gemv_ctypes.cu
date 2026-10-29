// Copyright (c) Nex-AGI. All rights reserved.
// Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// nex_n2_5_pro lm_head GEMV (one TP rank): logits_local[31040] = h @ W_local[31040,4096]^T
//
// Column/vocab-parallel: h (post model.norm) is REPLICATED [4096] bf16; W_local is this rank's vocab
// shard [31040,4096] bf16 (row-major). Each output logit is a complete dot over k=4096 -> NO cross-rank
// reduction. Streams the whole 254 MB shard once -> HBM-bandwidth-bound (~76 us wall @100% MBU).
//
// One CTA computes each row. CTA 0 also computes inv_rms while streaming h. W_fused contains the folded
// (1+norm_w) factor; inv_rms is applied downstream because positive scaling does not change top-k IDs.
// Output is FP32 (lm_head/sampler are precision-sensitive; the top-k consumes these logits directly).
//
// Two geometries write the same bits. Geometry 0 (released) runs one CTA per row. Geometry 1 (H20)
// runs kH20RowsPerCta rows per CTA and issues every load before the first multiply, so that an SM
// keeps more bytes in flight. The launcher selects geometry 1 on devices with fewer than 128 SMs;
// NEXRT_H20_GEOMETRY overrides it (see h20_geometry.cuh).
#ifndef NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
#define NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY bs1
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include "../common/h20_geometry.cuh"

namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY {
constexpr int kDim = 4096;
constexpr int kDimPairs = kDim / 2;  // 2048  (__nv_bfloat162 pairs)
constexpr int kRows = 31040;         // vocab / TP = 248320 / 8
constexpr float kInvDim = 1.0f / static_cast<float>(kDim);
static_assert(kDim % 2 == 0, "kDim must be even for bf16x2");
// One CTA per row + fused RMS in CTA 0 (raw logits + inv_rms; W is (1+norm_w)-folded).
template <int NT>
__global__ __launch_bounds__(NT) void LmHeadGemvRowRmsCta0Kernel(
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ Wf,
    float* __restrict__ logits, float* __restrict__ inv_rms, float eps) {
    const int row = blockIdx.x;
    if (row >= kRows) {
        return;
    }
    const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(h);
    const auto* __restrict__ w2 =
        reinterpret_cast<const __nv_bfloat162*>(Wf + static_cast<size_t>(row) * kDim);
    float acc = 0.0f, rms_acc = 0.0f;
    for (int p = threadIdx.x; p < kDimPairs; p += NT) {
        const float2 hf = __bfloat1622float2(h2[p]);
        const float2 wf = __bfloat1622float2(w2[p]);
        acc += hf.x * wf.x + hf.y * wf.y;
        if (row == 0) {
            rms_acc += hf.x * hf.x + hf.y * hf.y;
        }  // only CTA0 does the variance reduce
    }
    __shared__ typename cub::BlockReduce<float, NT>::TempStorage br_tmp;
    const float sum = cub::BlockReduce<float, NT>(br_tmp).Sum(acc);
    if (threadIdx.x == 0) {
        logits[row] = sum;
    }  // RAW (inv_rms applied downstream)
    if (row == 0) {
        __syncthreads();  // reuse br_tmp for the CTA0 variance reduce
        const float sq = cub::BlockReduce<float, NT>(br_tmp).Sum(rms_acc);
        if (threadIdx.x == 0) {
            inv_rms[0] = rsqrtf(sq * kInvDim + eps);
        }
    }
}

// The H20 geometry: kRowsPerCta consecutive rows per CTA, and each thread issues all its loads (the
// pairs of h, then the pairs of every row) before its first multiply. One row per CTA keeps 4 loads in
// flight per thread, too few to fill the memory pipe of an H20 (78 SMs). Every row keeps the released
// arithmetic: thread t sums the pairs t + NT * j in order j from 0.0f with the same expression, and the
// same BlockReduce sums the NT partials, so both geometries write the same bits. The kernel overloads
// the released kernel's name, so the MTP clone's rename and the kernel-name patterns of the graph tests
// cover it, and the released instantiation keeps its symbol and machine code. The bound of 1 CTA per
// SM lifts the register cap that ptxas otherwise sets for occupancy (32 registers, which moves 22 of
// the 40 loads behind multiplies): all loads come first, in 56 registers, at 4 CTAs per SM.
template <int NT, int kRowsPerCta>
__global__ __launch_bounds__(NT, 1) void LmHeadGemvRowRmsCta0Kernel(
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ Wf,
    float* __restrict__ logits, float* __restrict__ inv_rms, float eps) {
    constexpr int kPairsPerThread = kDimPairs / NT;
    static_assert(kDimPairs % NT == 0, "every thread sums the same number of pairs");
    static_assert(kRows % kRowsPerCta == 0, "the grid covers every row once");
    const int row0 = blockIdx.x * kRowsPerCta;
    if (row0 >= kRows) {
        return;
    }
    const auto* __restrict__ h2 = reinterpret_cast<const __nv_bfloat162*>(h);
    __nv_bfloat162 hv[kPairsPerThread];
    __nv_bfloat162 wv[kRowsPerCta][kPairsPerThread];
#pragma unroll
    for (int j = 0; j < kPairsPerThread; ++j) {
        hv[j] = h2[threadIdx.x + j * NT];
    }
#pragma unroll
    for (int r = 0; r < kRowsPerCta; ++r) {
        const auto* __restrict__ w2 =
            reinterpret_cast<const __nv_bfloat162*>(Wf + static_cast<size_t>(row0 + r) * kDim);
#pragma unroll
        for (int j = 0; j < kPairsPerThread; ++j) {
            wv[r][j] = w2[threadIdx.x + j * NT];
        }
    }
    float acc[kRowsPerCta];
#pragma unroll
    for (int r = 0; r < kRowsPerCta; ++r) {
        acc[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < kPairsPerThread; ++j) {
            const float2 hf = __bfloat1622float2(hv[j]);
            const float2 wf = __bfloat1622float2(wv[r][j]);
            acc[r] += hf.x * wf.x + hf.y * wf.y;
        }
    }
    __shared__ typename cub::BlockReduce<float, NT>::TempStorage br_tmp[kRowsPerCta];
#pragma unroll
    for (int r = 0; r < kRowsPerCta; ++r) {
        const float sum = cub::BlockReduce<float, NT>(br_tmp[r]).Sum(acc[r]);
        if (threadIdx.x == 0) {
            logits[row0 + r] = sum;
        }  // RAW (inv_rms applied downstream)
    }
    if (row0 == 0) {
        float rms_acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kPairsPerThread; ++j) {
            const float2 hf = __bfloat1622float2(hv[j]);
            rms_acc += hf.x * hf.x + hf.y * hf.y;
        }
        __syncthreads();  // reuse br_tmp[0] for the CTA0 variance reduce
        const float sq = cub::BlockReduce<float, NT>(br_tmp[0]).Sum(rms_acc);
        if (threadIdx.x == 0) {
            inv_rms[0] = rsqrtf(sq * kInvDim + eps);
        }
    }
}

template <int NT>
int LaunchRowRms(const void* h, const void* Wf, void* logits, float* inv_rms, float eps,
                 void* stream) {
    LmHeadGemvRowRmsCta0Kernel<NT><<<dim3(kRows), dim3(NT), 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(Wf),
        static_cast<float*>(logits), inv_rms, eps);
    return static_cast<int>(cudaGetLastError());
}

// Rows per CTA of the H20 geometry. Of 1, 2, 4, 5 and 8, 4 ran fastest in a microbenchmark on an H20
// (73 us against 103 us for the released geometry, with the L2 flushed before each launch).
constexpr int kH20RowsPerCta = 4;

template <int NT, int kRowsPerCta>
int LaunchRowsRms(const void* h, const void* Wf, void* logits, float* inv_rms, float eps,
                  void* stream) {
    LmHeadGemvRowRmsCta0Kernel<NT, kRowsPerCta>
        <<<dim3(kRows / kRowsPerCta), dim3(NT), 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16*>(h), static_cast<const __nv_bfloat16*>(Wf),
            static_cast<float*>(logits), inv_rms, eps);
    return static_cast<int>(cudaGetLastError());
}

// geometry 0 launches the released kernel and grid, geometry 1 the H20 geometry. Other values return
// cudaErrorInvalidValue without a launch.
inline int LaunchGeometry(const void* h, const void* Wf, void* logits, float* inv_rms, float eps,
                          int geometry, void* stream) {
    if (geometry == 0) {
        return LaunchRowRms<256>(h, Wf, logits, inv_rms, eps, stream);
    }
    if (geometry == 1) {
        return LaunchRowsRms<256, kH20RowsPerCta>(h, Wf, logits, inv_rms, eps, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}
}  // namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY
using namespace nexrt::sm90::nex_n2_5_pro::NEXRT_NEX_N2_5_PRO_KERNEL_FAMILY;
extern "C" {
// The geometry follows the device and NEXRT_H20_GEOMETRY (h20_geometry.cuh). The kernel reads only
// this rank's shard, so ranks need not agree.
int NexN25ProLmHeadGemvLaunch(const void* h, const void* Wf, void* o, float* r, float e, void* s) {
    const int geometry = h20_geometry();
    if (geometry < 0) return -geometry;
    return LaunchGeometry(h, Wf, o, r, e, geometry, s);
}
#ifndef NEXRT_MTP_RUNTIME_ONLY
// The geometry that NexN25ProLmHeadGemvLaunch uses on the current device: 0 or 1, or a negative CUDA
// error code (an invalid NEXRT_H20_GEOMETRY gives -cudaErrorInvalidValue). head.py checks it at start.
int NexN25ProLmHeadGemvGeometry() { return h20_geometry(); }

// NexN25ProLmHeadGemvLaunch with an explicit geometry: 0 = released, 1 = H20. Other values return
// cudaErrorInvalidValue without a launch.
int NexN25ProLmHeadGemvLaunchGeometry(const void* h, const void* Wf, void* o, float* r, float e,
                                      int geometry, void* s) {
    return LaunchGeometry(h, Wf, o, r, e, geometry, s);
}

const char* NexN25ProLmHeadCudaErrorString(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
#endif
}
