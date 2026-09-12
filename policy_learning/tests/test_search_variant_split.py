"""탐색 설정이 바뀌면 다른 팔로 센다 — 2026-09-12 회귀 방지.

그날 걸음을 2°→5°, min_gain 을 0.01→0.02 로 바꿨다 (dQ/dθ≈0.005/° 라 2° 가 정지 잡음
0.007 에 묻혔다). 전후를 한 팔로 묶으면 서로 다른 알고리즘의 성공률을 평균내게 된다.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_experiment import UNRECORDED, search_variant, split_search, variants_by_dir


def _ep(root: Path, name: str, condition: str, search=None):
    d = root / name
    d.mkdir(parents=True)
    meta = {"condition": condition, "episode_started": True}
    if search is not None:
        meta["search"] = search
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_variant_tag_from_meta():
    assert search_variant({"search": {"step_deg": 5.0, "min_gain": 0.02}}) == "5°/0.02"
    assert search_variant({"search": {"step_deg": 2.0, "min_gain": 0.01}}) == "2°/0.01"


def test_missing_settings_are_named_not_guessed():
    """옛 에피소드는 설정이 기록돼 있지 않다 — 값을 지어내지 않고 그렇게 표시한다."""
    assert search_variant({}) == UNRECORDED
    assert search_variant({"search": {"step_deg": 5.0}}) == UNRECORDED


def test_single_variant_is_left_alone(tmp_path):
    """설정이 한 가지뿐이면 이름을 어지럽히지 않는다."""
    _ep(tmp_path, "ep0001", "search", {"step_deg": 5.0, "min_gain": 0.02})
    _ep(tmp_path, "ep0002", "search", {"step_deg": 5.0, "min_gain": 0.02})
    v = variants_by_dir(tmp_path)
    assert split_search("search", "ep0001", v) == "search"


def test_two_variants_are_split(tmp_path):
    _ep(tmp_path, "ep0001", "search")                                   # 옛 것
    _ep(tmp_path, "ep0002", "search", {"step_deg": 5.0, "min_gain": 0.02})
    v = variants_by_dir(tmp_path)
    assert split_search("search", "ep0001", v) == f"search[{UNRECORDED}]"
    assert split_search("search", "ep0002", v) == "search[5°/0.02]"


def test_other_conditions_are_never_split(tmp_path):
    _ep(tmp_path, "ep0001", "search")
    _ep(tmp_path, "ep0002", "search", {"step_deg": 5.0, "min_gain": 0.02})
    _ep(tmp_path, "ep0003", "placebo")
    v = variants_by_dir(tmp_path)
    assert "ep0003" not in v
    for c in ("hold", "placebo", "policy"):
        assert split_search(c, "ep0003", v) == c
