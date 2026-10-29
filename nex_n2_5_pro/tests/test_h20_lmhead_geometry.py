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

"""LM head GEMV: the H20 geometry writes the same bits as the released geometry.

NexN25ProLmHeadGemvLaunchGeometry runs geometry 0 (released) and geometry 1 (H20) on the same
inputs, and both must write the same logits and inv_rms bits. NEXRT_H20_GEOMETRY selects the
geometry of the released entry NexN25ProLmHeadGemvLaunch: unset or empty selects automatically
(1 on fewer than 128 SMs, else 0), 0 and 1 force it, and any other value is an error. The
selection is fixed per process, so each setting runs in its own subprocess.

Run on one GPU from the repository root:
    python3 -m pytest -q nex_n2_5_pro/tests/test_h20_lmhead_geometry.py
"""

import ctypes
import json
import os
import subprocess
import sys

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from nex_n2_5_pro.head import DIM, EPS, ROWS
from nex_n2_5_pro.kernel_build import KERNEL_ROOT, REPO_ROOT, build_kernel

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
vp, cf, ci = ctypes.c_void_p, ctypes.c_float, ctypes.c_int
ptr = lambda tensor: vp(tensor.data_ptr())
stream = lambda: vp(torch.cuda.current_stream().cuda_stream)


def load_library():
    lib = ctypes.CDLL(str(build_kernel(KERNEL_ROOT / "head", "nex_n2_5_pro_lmhead_gemv_ctypes")))
    lib.NexN25ProLmHeadGemvLaunch.argtypes = [vp, vp, vp, vp, cf, vp]
    lib.NexN25ProLmHeadGemvLaunch.restype = ci
    lib.NexN25ProLmHeadGemvLaunchGeometry.argtypes = [vp, vp, vp, vp, cf, ci, vp]
    lib.NexN25ProLmHeadGemvLaunchGeometry.restype = ci
    lib.NexN25ProLmHeadGemvGeometry.argtypes = []
    lib.NexN25ProLmHeadGemvGeometry.restype = ci
    return lib


def run(launch, h, W, *geometry):
    """Launch into NaN-filled outputs and return (rc, logits, inv_rms)."""
    logits = torch.full((ROWS,), float("nan"), device=h.device)
    inv_rms = torch.full((1,), float("nan"), device=h.device)
    rc = launch(ptr(h), ptr(W), ptr(logits), ptr(inv_rms), cf(EPS), *geometry, stream())
    torch.cuda.synchronize()
    return rc, logits, inv_rms


def kernels(fn):
    """Return (fn's result, the sorted names of the CUDA kernels that fn launched)."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    names = [evt.key for evt in prof.key_averages() if evt.self_device_time_total > 0]
    return result, sorted(names)


@pytest.fixture(scope="module")
def lib():
    return load_library()


def inputs(seed, h_scale, w_scale, wide, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    h = torch.randn(DIM, generator=gen, device=device) * h_scale
    W = torch.randn(ROWS, DIM, generator=gen, device=device) * w_scale
    if wide:  # magnitudes spread over many binades, so a changed summation order changes bits
        W *= torch.exp2(torch.randn(ROWS, DIM, generator=gen, device=device) * 4)
    return h.to(torch.bfloat16), W.to(torch.bfloat16)


@needs_gpu
@pytest.mark.parametrize(
    "seed, h_scale, w_scale, wide",
    [(1, 1.0, 0.02, False), (2, 30.0, 0.02, True), (3, 1e-3, 1.0, False)],
)
def test_h20_geometry_matches_released_bitwise(lib, seed, h_scale, w_scale, wide):
    device = torch.device("cuda", 0)
    h, W = inputs(seed, h_scale, w_scale, wide, device)
    rc0, logits0, inv0 = run(lib.NexN25ProLmHeadGemvLaunchGeometry, h, W, 0)
    rc1, logits1, inv1 = run(lib.NexN25ProLmHeadGemvLaunchGeometry, h, W, 1)
    rc, logits, inv = run(lib.NexN25ProLmHeadGemvLaunch, h, W)
    assert (rc0, rc1, rc) == (0, 0, 0)

    # Every row is written, with the GEMV value (a check that the bit test below is not vacuous).
    hd, Wd = h.double(), W.double()
    bound = 1e-5 * (Wd.abs() @ hd.abs()) + 1e-30
    assert ((logits0.double() - Wd @ hd).abs() <= bound).all()
    inv_ref = (hd.square().mean() + EPS).rsqrt()
    assert abs(inv0.item() / inv_ref.item() - 1) < 1e-5

    for name, got in (("geometry 1", (logits1, inv1)), ("released entry", (logits, inv))):
        assert torch.equal(got[0].view(torch.int32), logits0.view(torch.int32)), name
        assert torch.equal(got[1].view(torch.int32), inv0.view(torch.int32)), name


@needs_gpu
def test_inv_rms_matches_released_bitwise(lib):
    """16 more h vectors for the CTA-0 variance reduce, one launch per geometry each."""
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(4)
    W = (torch.randn(ROWS, DIM, generator=gen, device=device) * 0.02).to(torch.bfloat16)
    for i in range(16):
        h = torch.randn(DIM, generator=gen, device=device) * 2.0 ** (i - 8)
        if i % 2:  # magnitudes spread over many binades
            h *= torch.exp2(torch.randn(DIM, generator=gen, device=device) * 4)
        h = h.to(torch.bfloat16)
        rc0, logits0, inv0 = run(lib.NexN25ProLmHeadGemvLaunchGeometry, h, W, 0)
        rc1, logits1, inv1 = run(lib.NexN25ProLmHeadGemvLaunchGeometry, h, W, 1)
        assert (rc0, rc1) == (0, 0) and torch.isfinite(inv0).all(), i
        assert torch.equal(inv1.view(torch.int32), inv0.view(torch.int32)), i
        assert torch.equal(logits1.view(torch.int32), logits0.view(torch.int32)), i


@needs_gpu
def test_other_geometry_values_are_refused(lib):
    device = torch.device("cuda", 0)
    h = torch.ones(DIM, dtype=torch.bfloat16, device=device)
    W = torch.ones(ROWS, DIM, dtype=torch.bfloat16, device=device)
    for geometry in (-1, 2):
        rc, logits, inv_rms = run(lib.NexN25ProLmHeadGemvLaunchGeometry, h, W, geometry)
        assert rc != 0, geometry
        assert logits.isnan().all() and inv_rms.isnan().all(), geometry


def child_report():
    """Body of the subprocess of test_environment_selects_the_geometry: print one JSON line."""
    from nex_n2_5_pro.head import lmhead_geometry

    lib = load_library()
    device = torch.device("cuda", 0)
    report = {
        "sms": torch.cuda.get_device_properties(device).multi_processor_count,
        "geometry": lib.NexN25ProLmHeadGemvGeometry(),
    }
    try:
        report["head_check"] = lmhead_geometry(lib)
    except RuntimeError as err:
        report["head_check"] = str(err)
    h = torch.zeros(DIM, dtype=torch.bfloat16, device=device)
    W = torch.zeros(ROWS, DIM, dtype=torch.bfloat16, device=device)
    logits = torch.empty(ROWS, device=device)
    inv_rms = torch.empty(1, device=device)
    args = (ptr(h), ptr(W), ptr(logits), ptr(inv_rms), cf(EPS))
    for key, launch, geometry in (
        ("entry", lib.NexN25ProLmHeadGemvLaunch, ()),
        ("geometry0", lib.NexN25ProLmHeadGemvLaunchGeometry, (0,)),
        ("geometry1", lib.NexN25ProLmHeadGemvLaunchGeometry, (1,)),
    ):
        report[key] = list(kernels(lambda: launch(*args, *geometry, stream())))
    print(json.dumps(report), flush=True)


@needs_gpu
@pytest.mark.parametrize("value", [None, "", "0", "1", "2", "h20"])
def test_environment_selects_the_geometry(value):
    env = {k: v for k, v in os.environ.items() if k != "NEXRT_H20_GEOMETRY"}
    if value is not None:
        env["NEXRT_H20_GEOMETRY"] = value
    code = "from nex_n2_5_pro.tests.test_h20_lmhead_geometry import child_report; child_report()"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])

    # The two geometries are different kernels, so the bitwise test compares two code paths.
    rc0, names0 = report["geometry0"]
    rc1, names1 = report["geometry1"]
    assert rc0 == 0 and rc1 == 0 and len(names0) == 1 and len(names1) == 1
    assert names0 != names1
    if value in ("2", "h20"):
        assert report["geometry"] < 0
        assert "(invalid argument)" in str(report["head_check"])  # the decoded CUDA error
        assert "NEXRT_H20_GEOMETRY" in str(report["head_check"])
        rc, names = report["entry"]
        assert rc != 0 and names == []  # the released entry refuses and launches nothing
    else:
        expected = int(value) if value else int(report["sms"] < 128)
        assert report["geometry"] == expected
        assert report["head_check"] == expected
        assert report["entry"] == [0, names1 if expected else names0]


@needs_gpu
def test_failed_query_reads_differently():
    """Without a visible GPU the query fails in cudaGetDevice, and head.py names that error."""
    env = {k: v for k, v in os.environ.items() if k != "NEXRT_H20_GEOMETRY"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "from nex_n2_5_pro.head import lmhead_geometry\n"
        "from nex_n2_5_pro.tests.test_h20_lmhead_geometry import load_library\n"
        "try:\n    print(lmhead_geometry(load_library()))\n"
        "except RuntimeError as err:\n    print(err)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    assert "failed with CUDA error" in out.stdout, out.stdout
    assert "(invalid argument)" not in out.stdout, out.stdout
