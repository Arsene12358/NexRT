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

"""Host-only check that every rank launches the same FusedL2 CTA count.

Run from the repository root:  python -m pytest -q nex_n2_5_pro/tests/test_fused_l2_uniform.py
"""

from nex_n2_5_pro.moe import check_uniform_ctas
import pytest


def test_uniform_value_is_returned():
    assert check_uniform_ctas([64] * 8) == 64


def test_mismatch_raises():
    with pytest.raises(RuntimeError, match="differs across ranks"):
        check_uniform_ctas([64] * 7 + [128])


def test_invalid_value_raises():
    with pytest.raises(RuntimeError, match="invalid FusedL2 CTA count"):
        check_uniform_ctas([96] * 8)
