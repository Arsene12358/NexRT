#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H20 geometry of the MoE gate/up, GDN front and CP front kernels: bitwise equal to the released one.

Each kernel has a second launch entry with an `int geometry` argument. Geometry 0 runs the released
grid. Geometry 1 runs the H20 grid, which gives each thread block more rows, so that the grid fits one
round of thread blocks on a GPU with 78 SMs. Both geometries compute each output row with the same
operations in the same order. So on the same random inputs they must give the same bits, and so must
the released entry point, which selects the geometry by itself.

NEXRT_H20_GEOMETRY forces the selection: unset or empty selects geometry 1 when the GPU has fewer than
128 SMs and geometry 0 otherwise, 0 or 1 forces that geometry, and any other value is an error. The
test checks the selection under the value that it runs with, and an invalid value in a child process.

No model weights required. Run on one GPU from the repository root:
    python3 -m nex_n2_5_pro.tests.test_h20_geometry
    NEXRT_H20_GEOMETRY=0 python3 -m nex_n2_5_pro.tests.test_h20_geometry
"""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import torch

from nex_n2_5_pro.kernel_build import build_kernel

KERNELS = Path(__file__).resolve().parents[1] / "kernels"
FP8 = torch.float8_e4m3fn
DIM = 4096
SEEDS = (0, 1, 2)
INVALID_VALUE = 1  # cudaErrorInvalidValue
VP, CI = ctypes.c_void_p, ctypes.c_int

# MoE gate/up: 10 routed experts plus the shared expert, 128 rows each, w13 stacked per expert.
N_EXPERTS, TOPK, ROWS_PER_EXPERT = 512, 10, 128
# GDN front: 1536 qkv rows and 1024 z rows in fp8, 8 b rows and 8 a rows in bf16.
QKV_ROWS, Z_ROWS, B_ROWS = 1536, 1024, 8
# CP front: 16 Q heads, 4 gate heads, 1 K head and 1 V head of 256 rows each.
CP_ROWS, HEAD_DIM, KV_POSITIONS = 5632, 256, 8

# (library, kernel directory, source stem, build flags as the engine builds it, released entries)
LIBRARIES = {
    "gate/up": ("moe", "nex_n2_5_pro_l1_pair_ctypes", {}, "NexN25ProL1W13Xn"),
    "GDN front": (
        "linear_attn",
        "nex_n2_5_pro_linear_front_gemv_ctypes",
        {},
        "NexN25ProLinearFront",
    ),
    "CP front": (
        "full_attn",
        "nex_n2_5_pro_cp_qgate_ctypes",
        {"hopper": True, "link_cuda": True},
        "NexN25ProCpQGate",
    ),
}


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def bits(t):
    """Raw bytes, so that the comparison also covers NaN payloads and signed zeros."""
    return t.contiguous().view(torch.uint8)


def fp8_weights(gen, *shape):
    """Random FP8 weights over the whole E4M3 range (no NaN), made in slices of 64 rows."""
    w = torch.empty(*shape, dtype=FP8, device="cuda")
    flat = w.view(-1, shape[-1])
    for i in range(0, flat.shape[0], 64 * 256):
        part = flat[i : i + 64 * 256]
        part.copy_((torch.randn(part.shape, generator=gen, device="cuda") * 96).clamp(-448, 448))
    return w


def block_scales(gen, *shape):
    return torch.rand(*shape, generator=gen, device="cuda") * 2e-3 + 1e-4


def load(name):
    kdir, stem, flags, prefix = LIBRARIES[name]
    lib = ctypes.CDLL(str(build_kernel(KERNELS / kdir, stem, **flags)))
    lib.query = getattr(lib, f"{prefix}Geometry")
    lib.query.argtypes, lib.query.restype = [], CI
    return lib


def run_three_ways(name, launch, outputs):
    """Run the released geometry, the H20 geometry and the automatic entry on the same inputs.
    launch(geometry) runs one launch (geometry None = the released entry point) and returns its rc.
    Each output starts as NaN, so a row that a geometry does not write fails the comparison."""
    results = []
    for geometry in (0, 1, None):
        for t in outputs():
            t.fill_(float("nan"))
        rc = launch(geometry)
        torch.cuda.synchronize()
        assert rc == 0, f"{name}: geometry {geometry} returned {rc}"
        results.append([t.clone() for t in outputs()])
    for t in results[0]:
        assert bool(torch.isfinite(t.float()).all()), f"{name}: the released geometry left a NaN"
    for label, other in (("H20", results[1]), ("automatic", results[2])):
        for i, (a, b) in enumerate(zip(results[0], other)):
            assert torch.equal(bits(a), bits(b)), (
                f"{name}: output {i} of the {label} geometry differs"
            )


def check_rejects_geometry_2(name, launch, outputs):
    """An explicit geometry other than 0 or 1 returns cudaErrorInvalidValue and launches nothing."""
    for t in outputs():
        t.fill_(-7.0)
    before = [t.clone() for t in outputs()]
    rc = launch(2)
    torch.cuda.synchronize()
    assert rc == INVALID_VALUE, f"{name}: geometry 2 returned {rc}"
    for a, b in zip(before, outputs()):
        assert torch.equal(bits(a), bits(b)), f"{name}: geometry 2 wrote an output"


def test_gate_up(lib, gen):
    explicit = lib.NexN25ProL1W13XnLaunchGeometry
    explicit.argtypes, explicit.restype = [VP] * 7 + [CI, VP], CI
    auto = lib.NexN25ProL1W13XnLaunch
    auto.argtypes, auto.restype = [VP] * 8, CI
    w13 = fp8_weights(gen, N_EXPERTS, 2 * ROWS_PER_EXPERT, DIM)
    shared = fp8_weights(gen, 2 * ROWS_PER_EXPERT, DIM)
    raw_h = torch.empty((TOPK + 1) * ROWS_PER_EXPERT, dtype=torch.float32, device="cuda")
    for seed in SEEDS:
        gen.manual_seed(1000 + seed)
        w13_scales = block_scales(gen, N_EXPERTS, 2 * DIM // 128)
        shared_scales = block_scales(gen, 2 * DIM // 128)
        xn = torch.randn(DIM, generator=gen, device="cuda").to(torch.bfloat16)
        packed = torch.zeros(64, dtype=torch.int32, device="cuda")
        packed[:TOPK] = torch.randperm(N_EXPERTS, generator=gen, device="cuda")[:TOPK].int()
        packed[63] = torch.tensor([0.37], dtype=torch.float32).view(torch.int32).item()
        args = [ptr(t) for t in (packed, xn, w13, w13_scales, shared, shared_scales, raw_h)]

        def launch(geometry):
            if geometry is None:
                return auto(*args, stream())
            return explicit(*args, geometry, stream())

        run_three_ways("gate/up", launch, lambda: [raw_h])
    check_rejects_geometry_2("gate/up", launch, lambda: [raw_h])
    print(f"PASS gate/up: {len(SEEDS)} seeds, geometry 0 == geometry 1 == automatic", flush=True)


def test_gdn_front(lib, gen):
    explicit = lib.NexN25ProLinearFrontLaunchGeometry
    explicit.argtypes, explicit.restype = [VP] * 12 + [CI, VP], CI
    auto = lib.NexN25ProLinearFrontLaunch
    auto.argtypes, auto.restype = [VP] * 13, CI
    fp8_rows = QKV_ROWS + Z_ROWS
    qkvz_w = fp8_weights(gen, fp8_rows, DIM)
    qkv_out = torch.empty(QKV_ROWS, dtype=torch.bfloat16, device="cuda")
    z_out = torch.empty(Z_ROWS, dtype=torch.bfloat16, device="cuda")
    beta_out = torch.empty(B_ROWS, dtype=torch.float32, device="cuda")
    g_out = torch.empty(B_ROWS, dtype=torch.float32, device="cuda")
    state = torch.empty(QKV_ROWS, 4, dtype=torch.bfloat16, device="cuda")
    for seed in SEEDS:
        gen.manual_seed(2000 + seed)
        qkvz_scales = block_scales(gen, fp8_rows // 128, DIM // 128)
        b_w = (torch.randn(B_ROWS, DIM, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
        a_w = (torch.randn(B_ROWS, DIM, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
        a_log_dt = torch.randn(2 * B_ROWS, generator=gen, device="cuda") * 0.5
        conv_w = (torch.randn(QKV_ROWS, 4, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        state0 = torch.randn(QKV_ROWS, 4, generator=gen, device="cuda").to(torch.bfloat16)
        # Three decode steps: the conv state that each step writes feeds the next step.
        xs = [torch.randn(DIM, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(3)]
        weights = [ptr(t) for t in (qkvz_w, qkvz_scales, b_w, a_w, a_log_dt, conv_w)]
        outs = [ptr(t) for t in (state, qkv_out, z_out, beta_out, g_out)]

        def launch(geometry, x):
            args = [ptr(x), *weights, *outs]
            if geometry is None:
                return auto(*args, stream())
            return explicit(*args, geometry, stream())

        results = []
        for geometry in (0, 1, None):
            state.copy_(state0)
            steps = []
            for x in xs:
                for t in (qkv_out, z_out, beta_out, g_out):
                    t.fill_(float("nan"))
                rc = launch(geometry, x)
                torch.cuda.synchronize()
                assert rc == 0, f"GDN front: geometry {geometry} returned {rc}"
                steps.append([t.clone() for t in (state, qkv_out, z_out, beta_out, g_out)])
            results.append(steps)
        for step in results[0]:
            for t in step:
                assert bool(torch.isfinite(t.float()).all()), "GDN front: released geometry NaN"
        for label, other in (("H20", results[1]), ("automatic", results[2])):
            for s, (a_step, b_step) in enumerate(zip(results[0], other)):
                for i, (a, b) in enumerate(zip(a_step, b_step)):
                    assert torch.equal(bits(a), bits(b)), (
                        f"GDN front: step {s} output {i} of the {label} geometry differs"
                    )
    check_rejects_geometry_2(
        "GDN front",
        lambda geometry: launch(geometry, xs[0]),
        lambda: [state, qkv_out, z_out, beta_out, g_out],
    )
    print(
        f"PASS GDN front: {len(SEEDS)} seeds x 3 steps, geometry 0 == geometry 1 == automatic",
        flush=True,
    )


def test_cp_front(lib, gen):
    fixed_explicit = lib.NexN25ProCpQGateLaunchGeometry
    fixed_explicit.argtypes, fixed_explicit.restype = [VP] * 7 + [CI, CI, VP], CI
    fixed_auto = lib.NexN25ProCpQGateLaunch
    fixed_auto.argtypes, fixed_auto.restype = [VP] * 7 + [CI, VP], CI
    dyn_explicit = lib.NexN25ProCpQGateDynLaunchGeometry
    dyn_explicit.argtypes, dyn_explicit.restype = [VP] * 8 + [CI, VP], CI
    dyn_auto = lib.NexN25ProCpQGateDynLaunch
    dyn_auto.argtypes, dyn_auto.restype = [VP] * 9, CI
    w = fp8_weights(gen, CP_ROWS, DIM)
    q_group = torch.empty(16 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    gate = torch.empty(4 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty(KV_POSITIONS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.empty(KV_POSITIONS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    outputs = lambda: [q_group, gate, k_cache, v_cache]  # noqa: E731
    for seed in SEEDS:
        gen.manual_seed(3000 + seed)
        scales = block_scales(gen, CP_ROWS // 128, DIM // 128)
        x = torch.randn(DIM, generator=gen, device="cuda").to(torch.bfloat16)
        slot = torch.tensor([123, 6], dtype=torch.int32, device="cuda")
        args = [ptr(t) for t in (x, w, scales, q_group, gate, k_cache, v_cache)]

        def fixed(geometry):
            if geometry is None:
                return fixed_auto(*args, 5, stream())
            return fixed_explicit(*args, 5, geometry, stream())

        def dyn(geometry):
            if geometry is None:
                return dyn_auto(*args, ptr(slot), stream())
            return dyn_explicit(*args, ptr(slot), geometry, stream())

        # The caches start as NaN, so the positions that a launch must not write stay NaN.
        for form, launch, pos in (("fixed", fixed, 5), ("device-S", dyn, 6)):
            run_three_ways_cp(f"CP front {form}", launch, outputs, pos)
    check_rejects_geometry_2("CP front fixed", fixed, outputs)
    check_rejects_geometry_2("CP front device-S", dyn, outputs)
    print(
        f"PASS CP front: {len(SEEDS)} seeds, fixed and device-S forms, "
        "geometry 0 == geometry 1 == automatic",
        flush=True,
    )


def run_three_ways_cp(name, launch, outputs, pos):
    """run_three_ways for the CP front, whose K and V caches are written only at one position."""
    results = []
    for geometry in (0, 1, None):
        for t in outputs():
            t.fill_(float("nan"))
        rc = launch(geometry)
        torch.cuda.synchronize()
        assert rc == 0, f"{name}: geometry {geometry} returned {rc}"
        results.append([t.clone() for t in outputs()])
    q_group, gate, k_cache, v_cache = results[0]
    for t in (q_group, gate, k_cache[pos], v_cache[pos]):
        assert bool(torch.isfinite(t.float()).all()), f"{name}: the released geometry left a NaN"
    others = torch.ones(KV_POSITIONS, dtype=torch.bool)
    others[pos] = False
    assert bool(torch.isnan(k_cache[others].float()).all()), f"{name}: wrote another position"
    assert bool(torch.isnan(v_cache[others].float()).all()), f"{name}: wrote another position"
    for label, other in (("H20", results[1]), ("automatic", results[2])):
        for i, (a, b) in enumerate(zip(results[0], other)):
            assert torch.equal(bits(a), bits(b)), (
                f"{name}: output {i} of the {label} geometry differs"
            )


CHILD = r"""
import ctypes, sys
import torch
vp, ci = ctypes.c_void_p, ctypes.c_int
entries = (  # (library index, query, released entry point, its argument types)
    (0, "NexN25ProL1W13XnGeometry", "NexN25ProL1W13XnLaunch", [vp] * 8),
    (1, "NexN25ProLinearFrontGeometry", "NexN25ProLinearFrontLaunch", [vp] * 13),
    (2, "NexN25ProCpQGateGeometry", "NexN25ProCpQGateLaunch", [vp] * 7 + [ci, vp]),
    (2, "NexN25ProCpQGateGeometry", "NexN25ProCpQGateDynLaunch", [vp] * 9),
)
libs = [ctypes.CDLL(path) for path in sys.argv[1:]]
codes = []
for index, query_name, launch_name, argtypes in entries:
    query = getattr(libs[index], query_name)
    query.restype = ci
    launch = getattr(libs[index], launch_name)
    launch.argtypes, launch.restype = argtypes, ci
    # Null pointers: a launch would fault, so rc 1 and a clean synchronize mean that none ran.
    codes.append((query(), launch(*[0 if t is ci else None for t in argtypes])))
torch.cuda.synchronize()
print("CODES", codes)
"""


def check_selection(libs):
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    env = os.environ.get("NEXRT_H20_GEOMETRY", "")
    assert env in ("", "0", "1"), f"run the test with NEXRT_H20_GEOMETRY unset, 0 or 1, not {env!r}"
    want = int(env) if env else (1 if sms < 128 else 0)
    for name, lib in libs.items():
        got = lib.query()
        assert got == want, (
            f"{name}: selected geometry {got}, expected {want} ({sms} SMs, env {env!r})"
        )
    # An invalid value: every query returns -cudaErrorInvalidValue, and the released entry points
    # return cudaErrorInvalidValue without a launch.
    paths = [
        str(build_kernel(KERNELS / kdir, stem, **flags))
        for kdir, stem, flags, _ in LIBRARIES.values()
    ]
    child = subprocess.run(
        [sys.executable, "-c", CHILD, *paths],
        env=dict(os.environ, NEXRT_H20_GEOMETRY="2"),
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, f"child failed: {child.stdout}{child.stderr}"
    codes = [line for line in child.stdout.splitlines() if line.startswith("CODES")]
    assert codes == [f"CODES {[(-INVALID_VALUE, INVALID_VALUE)] * 4}"], child.stdout
    print(
        f"PASS selection: geometry {want} on {sms} SMs with NEXRT_H20_GEOMETRY={env!r}, "
        "and NEXRT_H20_GEOMETRY=2 is an error that launches nothing",
        flush=True,
    )


def main():
    torch.cuda.set_device(0)
    libs = {name: load(name) for name in LIBRARIES}
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261029)
    test_gate_up(libs["gate/up"], gen)
    test_gdn_front(libs["GDN front"], gen)
    test_cp_front(libs["CP front"], gen)
    check_selection(libs)
    print("PASS H20 geometry: 3 kernels bitwise equal in both geometries", flush=True)


if __name__ == "__main__":
    main()
