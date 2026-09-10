"""두 부채꼴 변환 사본이 같은 출력을 내는지 고정한다.

`fr5_vision/scan_convert.py` 와 `imu_bench/host/us_scan_convert.py` 는 같은 코드지만
실행 환경이 갈려 두 벌로 둔다 (전자의 docstring 참조). 갈라지면 화면 기하와 저장
데이터의 기하가 조용히 달라지므로, 같은 입력에 같은 출력이 나오는 것을 여기서 잡는다.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "imu_bench", "host", "us_scan_convert.py")
)


def _load_sibling():
    """imu_bench 쪽 사본을 파일 경로로 직접 불러온다 (패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location("_imu_bench_scan_convert", _SIBLING)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # `dataclass` 는 annotation 을 풀 때 `sys.modules[cls.__module__]` 를 본다.
    # exec_module 전에 등록하지 않으면 그 조회가 None 이 되어 AttributeError 로 죽는다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not os.path.exists(_SIBLING), reason="imu_bench 사본이 없다")
def test_fan_output_identical():
    """무작위 극좌표 프레임에 대해 두 구현의 출력이 바이트 단위로 같아야 한다."""
    from fr5_vision import scan_convert as ours

    theirs = _load_sibling()
    assert theirs is not None

    rng = np.random.default_rng(0)
    polar = (rng.random((160, 512)) * 255).astype(np.uint8)

    for geo_kwargs in (
        {},
        {"radius_mm": 59.0, "half_angle_deg": 28.0, "depth_mm": 220.0},
        {"radius_mm": 40.0, "half_angle_deg": 35.0, "depth_mm": 150.0, "flip_lines": True},
    ):
        a = ours.ScanConverter(ours.FanGeometry(**geo_kwargs), 160, 512)
        b = theirs.ScanConverter(theirs.FanGeometry(**geo_kwargs), 160, 512)
        assert (a.out_h, a.out_w) == (b.out_h, b.out_w)
        assert a.mm_per_px == pytest.approx(b.mm_per_px)
        np.testing.assert_array_equal(a.convert(polar), b.convert(polar))
        assert a.meta() == b.meta()
