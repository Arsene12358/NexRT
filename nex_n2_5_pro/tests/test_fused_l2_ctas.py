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

"""TP8 FusedL2 reduce: the 64-block variant matches the 128-block variant bit for bit.

Both variants run through NexN25ProFusedL2ReduceResidLaunchCtas, in the plain form (x_resid NULL,
fp32 out), the fused-residual form, and the fused-residual form with the wait-shadow prefetch that
the decode engine passes (the kWaitPf instantiations), under three arrival skews. A captured
64-block launch then replays 10,000 times with alternating inputs, and the per-block epochs must
advance by 10,000.

No model weights required. Run on the GPU host from the repository root:
    torchrun --standalone --nproc_per_node=8 -m nex_n2_5_pro.tests.test_fused_l2_ctas
"""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as symm_mem

from nex_n2_5_pro.kernel_build import build_kernel

DIM, N_EXP, TOPK, RINTER = 4096, 512, 10, 128
FP8 = torch.float8_e4m3fn
REPLAYS = 10_000
PF_BYTES = 4 << 20  # the decode engine clamps the next layer's prefetch range to 4 MiB


def main():
    # The first check covers the automatic selection, so no override may apply.
    os.environ.pop("NEXRT_FUSED_L2_CTAS", None)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world == 8
    device = torch.device("cuda", torch.cuda.current_device())
    kdir = Path(__file__).resolve().parents[1] / "kernels" / "moe"
    lib = ctypes.CDLL(str(build_kernel(kdir, "nex_n2_5_pro_fused_l2_reduce_ctypes")))
    vp, cu, ci, cll = ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_longlong
    lib.NexN25ProFusedL2ReduceCtas.argtypes = []
    lib.NexN25ProFusedL2ReduceCtas.restype = ci
    launch_ctas = lib.NexN25ProFusedL2ReduceResidLaunchCtas
    launch_ctas.argtypes = [vp] * 7 + [cu, vp, cu] + [vp] * 4 + [ci, ci, vp, vp, cll, ci, vp]
    launch_ctas.restype = ci

    # The Moe.__init__ workspace: rep = replicated push buffer (written through its multicast
    # address), bar = one NVLS arrival slot per block, it_ctr = per-block epochs.
    rep = symm_mem.empty(8 * 2 * DIM, dtype=torch.float32, device=device)
    rep_handle = symm_mem.rendezvous(rep, dist.group.WORLD.group_name)
    bar = symm_mem.empty(128, dtype=torch.int32, device=device)
    bar_handle = symm_mem.rendezvous(bar, dist.group.WORLD.group_name)
    rep_mc, bar_mc = int(rep_handle.multicast_ptr), int(bar_handle.multicast_ptr)
    assert rep_mc and bar_mc, "the FusedL2 reduce requires NVLS multicast"
    rep.zero_()
    bar.zero_()
    it_ctr = torch.zeros(128, dtype=torch.int32, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.empty(DIM, dtype=torch.float32, device=device)
    resid = torch.empty(DIM, dtype=torch.bfloat16, device=device)

    # Weights and activations differ per rank, like the intermediate shards.
    gen = torch.Generator(device=device).manual_seed(1000 + rank)
    W2 = torch.randn(N_EXP, DIM, RINTER, device=device, generator=gen).to(FP8)
    w2_scales = torch.rand(N_EXP, DIM // 128, device=device, generator=gen) + 0.5
    shared_W2 = torch.randn(DIM, RINTER, device=device, generator=gen).to(FP8)
    shared_w2_scales = torch.rand(DIM // 128, device=device, generator=gen) + 0.5
    h_sets = [torch.randn(TOPK + 1, RINTER, device=device, generator=gen) for _ in range(2)]
    h = h_sets[0].clone()
    # Routing and the residual stream are replicated, so they are the same on every rank.
    common = torch.Generator().manual_seed(7)
    topk_ids = torch.randperm(N_EXP, generator=common)[:TOPK].to(torch.int32).to(device)
    topk_weights = torch.softmax(torch.randn(TOPK, generator=common), 0).to(device)
    x0 = torch.randn(DIM, generator=common).to(torch.bfloat16).to(device)
    torch.cuda.synchronize()
    dist.barrier()

    ptr = lambda tensor: vp(tensor.data_ptr())

    def launch(form, ctas):
        rc = launch_ctas(
            ptr(topk_ids),
            ptr(topk_weights),
            ptr(h),
            ptr(W2),
            ptr(w2_scales),
            ptr(shared_W2),
            ptr(shared_w2_scales),
            cu(rep_mc),
            ptr(rep),
            cu(bar_mc),
            ptr(bar),
            ptr(out),
            ptr(status),
            ptr(it_ctr),
            rank,
            world,
            ptr(resid) if form != "plain" else None,
            ptr(W2) if form == "resid_pf" else None,
            PF_BYTES if form == "resid_pf" else 0,
            ctas,
            vp(torch.cuda.current_stream().cuda_stream),
        )
        if rc:
            raise RuntimeError(f"FusedL2 launch failed: {rc}")

    def run(form, ctas, skew):
        out.fill_(float("nan"))  # rows that no block writes must not pass on stale values
        resid.copy_(x0)
        # Change the peer arrival order without changing any arithmetic input.
        torch.cuda._sleep(((rank + skew) % 8) * 10000)
        launch(form, ctas)
        torch.cuda.synchronize()
        # The variants map row pairs to different blocks, so the per-block parity does not keep
        # consecutive launches of different variants apart. Every rank finishes first.
        dist.barrier()
        return (out if form == "plain" else resid).clone()

    def all_ranks_bitwise_equal(tensor):
        gathered = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor)
        first = gathered[0].view(torch.uint8)
        return all(torch.equal(other.view(torch.uint8), first) for other in gathered)

    failures = 0

    def check(name, ok):
        nonlocal failures
        if not ok:
            failures += 1
            print(f"[rank {rank}] FAIL: {name}", flush=True)

    sm = torch.cuda.get_device_properties(device).multi_processor_count
    ctas = lib.NexN25ProFusedL2ReduceCtas()
    check(f"automatic CTA count {ctas} on {sm} SMs", ctas == (128 if sm >= 128 else 64))
    check(f"{ctas} blocks at one block per SM on {sm} SMs", ctas <= sm)

    first = {}
    for form in ("plain", "resid", "resid_pf"):
        for skew in (0, 3, 7):
            a, b = run(form, 128, skew), run(form, 64, skew)
            tag = f"{form} skew {skew}"
            check(f"{tag}: 64 equals 128", torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            check(f"{tag}: ranks equal", all_ranks_bitwise_equal(b))
            check(f"{tag}: status {status.item()}", status.item() == 0)
            first.setdefault(form, b)
            check(
                f"{tag}: equals skew 0",
                torch.equal(b.view(torch.uint8), first[form].view(torch.uint8)),
            )
    plain_out, resid_out = first["plain"], first["resid"]

    # Torch reference: this rank's fp32 partial, rounded to bf16 for the wire, then the ranks
    # summed in order r0..r7.
    acc = torch.zeros(DIM, device=device)
    for j in range(TOPK):
        e = int(topk_ids[j])
        w = W2[e].float() * w2_scales[e].repeat_interleave(128).view(DIM, 1)
        acc += topk_weights[j] * (w @ h[j])
    sw = shared_W2.float() * shared_w2_scales.repeat_interleave(128).view(DIM, 1)
    acc += sw @ h[TOPK]
    partials = [torch.empty_like(acc) for _ in range(world)]
    dist.all_gather(partials, acc)
    reference = torch.zeros(DIM, device=device)
    for partial in partials:
        reference += partial.to(torch.bfloat16).float()
    rel = ((plain_out - reference).norm() / reference.norm()).item()
    check(f"plain vs torch reference: relL2 {rel:.3e}", rel < 1e-3)
    check(
        "resid equals bf16(x0 + plain)",
        torch.equal(resid_out, (x0.float() + plain_out).to(torch.bfloat16)),
    )
    check(
        "resid with prefetch equals resid",
        torch.equal(first["resid_pf"].view(torch.uint8), resid_out.view(torch.uint8)),
    )

    # Graph: capture the 64-block plain launch once and replay it 10,000 times. The input
    # alternates between two sets, so a block that reads the other parity (the previous replay's
    # values) fails the per-replay check.
    eager = []
    for h_set in h_sets:
        h.copy_(h_set)
        eager.append(run("plain", 64, 0))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch("plain", 64)
    torch.cuda.synchronize()
    dist.barrier()
    it_before = it_ctr.clone()
    mismatches = torch.zeros((), dtype=torch.int32, device=device)
    for i in range(REPLAYS):
        h.copy_(h_sets[i & 1])
        graph.replay()
        mismatches += (out.view(torch.int32) != eager[i & 1].view(torch.int32)).any()
    torch.cuda.synchronize()
    graph_out, eager_out = out.clone(), eager[(REPLAYS - 1) & 1]
    check(
        f"graph equals eager, status {status.item()}",
        torch.equal(graph_out, eager_out) and status.item() == 0,
    )
    check(f"replays that differ from eager: {mismatches.item()}", mismatches.item() == 0)
    advanced = it_ctr[:64] - it_before[:64]
    expected = torch.full((64,), REPLAYS, dtype=torch.int32, device=device)
    check("it_ctr[:64] advanced by 10,000", torch.equal(advanced, expected))
    check("it_ctr[64:] unchanged", torch.equal(it_ctr[64:], it_before[64:]))

    result = torch.tensor([failures], device=device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    failures = int(result.item())
    if rank == 0:
        print(
            f"FusedL2 CTA variants: automatic {ctas} on {sm} SMs, 18 eager runs, relL2 {rel:.3e}, "
            f"{REPLAYS} graph replays, failures={failures}",
            flush=True,
        )
    dist.destroy_process_group()
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
