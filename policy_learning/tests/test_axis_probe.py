"""축 흔들기 — 규칙 기반 정렬에 필요한 **기하 이득**을 재는 모드.

왜 따로 재는가: 프리핸드의 centroid_dx↔θy 기울기 43 °/단위 는 "치우침이 이만큼일 때
시연자가 이만큼 돌렸다" 는 **행동** 기울기다. 제어기에 필요한 것은 "1° 돌리면 centroid 가
얼마나 움직이나" 이고 둘은 다른 양이다 (2026-09-12).
"""

import numpy as np
import pytest

from rus_policy.axis_probe import AXIS_INDEX, DEFAULT_ORDER, MOVE, REST, AxisProbe
from rus_policy.episode import CONDITIONS


def test_axis_probe_is_a_known_condition():
    assert "axis_probe" in CONDITIONS


def test_one_axis_at_a_time():
    """두 축이 동시에 돌면 이득이 섞여 분리되지 않는다 — 이 모드의 존재 이유다."""
    p = AxisProbe()
    for t in np.arange(0, p.total_s, 0.1):
        vel, _ = p.update(float(t))
        assert np.count_nonzero(vel) <= 1, (t, vel)


def test_every_axis_gets_both_signs():
    p = AxisProbe()
    seen = {}
    for t in np.arange(0, p.total_s, 0.05):
        vel, lab = p.update(float(t))
        if p.phase == MOVE:
            seen.setdefault(lab[:-1], set()).add(lab[-1])
    assert set(seen) == set(DEFAULT_ORDER)
    for name, signs in seen.items():
        assert signs == {"+", "−"}, (name, signs)


def test_rest_between_moves():
    """쉬는 구간이 있어야 시야 자체의 느린 변동을 기준선으로 뺄 수 있다."""
    p = AxisProbe()
    phases = [p.update(float(t))[0] for t in np.arange(0, p.slot_s * 2, 0.1)]
    moving = [bool(np.any(v)) for v in phases]
    assert True in moving and False in moving


def test_amplitude_follows_the_axis_kind():
    p = AxisProbe(amp_deg=3.0, amp_mm_s=5.0)
    got = {}
    for t in np.arange(0, p.total_s, 0.05):
        vel, lab = p.update(float(t))
        if p.phase == MOVE:
            got[lab[:-1]] = float(np.abs(vel).max())
    for name, v in got.items():
        assert v == pytest.approx(3.0 if name.startswith("th") else 5.0), (name, v)


def test_stops_after_the_schedule():
    """첫 호출이 기준 시각이 된다 — 순서를 다 돌면 0 을 내고 done 을 붙인다."""
    p = AxisProbe()
    p.update(0.0)
    vel, lab = p.update(p.total_s + 1.0)
    assert lab == "done" and not np.any(vel)


def test_beam_translation_is_never_commanded():
    """빔축 병진은 admittance 몫이라 이 모드가 건드리면 안 된다."""
    assert 2 not in AXIS_INDEX.values()
    p = AxisProbe()
    for t in np.arange(0, p.total_s, 0.1):
        assert p.update(float(t))[0][2] == 0.0


def test_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="모르는 축"):
        AxisProbe(order=("thx", "롤"))


# --- 기록이 실제로 남는가 (2026-09-12 회귀 방지) ------------------------------

def test_axis_probe_does_not_wait_for_the_start_gate():
    """측정 모드는 시작 조건을 기다리면 안 된다.

    2026-09-12: 게이트가 안 열려 60 s 를 흔들고도 "시도가 시작되지 않았다 — 기록하지
    않는다" 로 끝났다. 실험 조건은 게이트가 장치의 일부지만 측정은 아니다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    line = re.search(r"self\.gate_required = .*", src).group(0)
    assert "axis_probe" in line, line


def test_loop_mode_saves_when_the_duration_ends():
    """--loop 의 종료 경로는 summary() 뿐이라 기록이 사라졌다 — 시간 종료도 닫아야 한다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    block = re.search(r"if self\.args\.duration > 0 and \(t - self\.t_start\).*?return",
                      src, re.S).group(0)
    assert "_close_attempt()" in block, block


def test_loop_mode_saves_when_the_operator_interrupts():
    """Ctrl-C 도 한 시도의 끝이다.

    2026-09-14 rule_v2: 규칙 제어기를 3 분 돌리고 Ctrl-C 로 끊었더니 force_search.log 만
    남았다. finally 가 summary() 만 불러서 열려 있던 시도가 통째로 사라진다. 조작자가
    Stop 을 누르지 않고 터미널을 끊는 것이 오히려 흔한 경로다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    block = re.search(r"finally:.*?node\.destroy_node\(\)", src, re.S).group(0)
    loop = re.search(r"if args\.loop:.*?else:", block, re.S).group(0)
    assert "_close_attempt()" in loop, loop


def test_analyzer_finds_loop_mode_folders():
    """--loop 은 attempt_*, run_experiment 는 ep* 로 저장한다 — 둘 다 찾아야 한다."""
    import json as _json
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_axis_probe import resolve_episode
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("attempt_001", "ep0001_pose01_axis_probe"):
            d = root / name
            d.mkdir()
            (d / "meta.json").write_text(_json.dumps({"condition": "axis_probe"}), encoding="utf-8")
        assert resolve_episode(root) is not None


def test_signs_alternate_so_the_probe_stays_near_the_start():
    """같은 부호를 몰아서 하면 반복을 늘릴수록 시작 자세에서 멀어진다.

    repeats=3 · 12.5° 걸음이면 몰아서 할 때 한쪽으로 37.5°, 번갈아 하면 12.5° 다.
    """
    p = AxisProbe(repeats=3, move_s=2.5, amp_deg=5.0)
    seq, last = [], None
    for t in np.arange(0, p.total_s, 0.05):
        _, lab = p.update(float(t))
        if p.phase == MOVE and lab != last:
            seq.append(lab)
            last = lab
    step = p.amp_deg * p.move_s
    acc, peak = {}, {}
    for lab in seq:
        name, sign = lab[:-1], (1.0 if lab[-1] == "+" else -1.0)
        acc[name] = acc.get(name, 0.0) + sign * step
        peak[name] = max(peak.get(name, 0.0), abs(acc[name]))
    for name, v in peak.items():
        assert v <= step + 1e-6, (name, v, step)


def test_waits_until_the_bladder_is_visible():
    """안 보이는 채로 흔들면 면적도 중심도 0 이라 어떤 이득도 안 나온다.

    2026-09-12·14 에 세 번 연속 그렇게 날렸다 — 앞 60 s 동안 마스크 0 % 였고 회전 세 축이
    통째로 거기 들어갔다.
    """
    p = AxisProbe(start_area=0.01, start_hold_s=1.0)
    t = 0.0
    for _ in range(20):                       # 안 보인다 — 아무것도 안 한다
        vel, lab = p.update(t, 0.0)
        assert lab == "wait" and not np.any(vel)
        t += 0.5
    p.update(t, 0.02); t += 0.5               # 보이기 시작
    vel, lab = p.update(t, 0.02)
    assert lab == "wait", "유지 시간을 안 채우고 시작했다"
    t += 0.6
    vel, lab = p.update(t, 0.02)
    assert lab != "wait" and np.any(vel), "보이는데도 시작하지 않았다"


def test_visibility_hold_resets_if_the_bladder_flickers():
    """잠깐 보였다 사라지면 처음부터 다시 센다 — 문턱에 걸친 채로 시작하면 안 된다."""
    p = AxisProbe(start_area=0.01, start_hold_s=1.0)
    t = 0.0
    for a in (0.02, 0.02, 0.0, 0.02, 0.02):
        _, lab = p.update(t, a)
        assert lab == "wait", (t, a, lab)
        t += 0.4


def test_no_area_given_means_start_immediately():
    """면적을 안 주면 예전처럼 곧바로 시작한다 (단위 시험·합성 세션용)."""
    p = AxisProbe()
    vel, lab = p.update(0.0)
    assert lab != "wait" and np.any(vel)


def test_attempt_numbering_continues_past_existing_folders():
    """같은 --out 을 다시 쓰면 앞의 기록을 덮으면 안 된다.

    2026-09-14: rule_v2 에서 123 s 를 유지한 attempt_001 이 다음 실행에 지워졌다.
    시도 번호가 프로세스마다 0 에서 다시 시작했기 때문이다. 한 자세에서 여러 번 받는
    것이 정상 운용이므로, 폴더 이름을 매번 바꾸라고 요구하는 쪽이 틀렸다.
    """
    import sys as _sys
    import tempfile
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_policy import _last_attempt
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert _last_attempt(root) == 0
        assert _last_attempt(root / "없는폴더") == 0
        for name in ("attempt_001", "attempt_002", "attempt_007"):
            (root / name).mkdir()
        (root / "attempt_bad").mkdir()          # 번호가 아닌 것은 무시한다
        assert _last_attempt(root) == 7


def test_meta_records_which_code_produced_it():
    """에피소드마다 코드 판본이 남아야 한다.

    2026-09-14 하루에 규칙 제어기가 일곱 번 바뀌었고 13 개가 쌓였는데, 파일만 보고는
    어느 것이 어느 코드였는지 알 수 없었다. 서로 다른 제어기를 평균 내게 된다.
    """
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_policy import code_provenance
    got = code_provenance()
    assert set(got) == {"git_sha", "git_dirty", "git_described"}
    assert isinstance(got["git_dirty"], bool)
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    assert '"code": code_provenance(),' in src, "meta 에 안 넣고 있다"


def test_default_paths_do_not_depend_on_the_working_directory():
    """cwd 가 어디든 같아야 한다.

    2026-09-14 본 시험 ep0002: 기본값이 "policy_learning/configs/…" 라 저장소 뿌리에서
    띄울 때만 맞았다. 런처는 policy_learning 안에서 러너를 띄우므로 그 자리에서 터졌고,
    드라이버는 그것을 "시작 조건 미충족 — 폐기" 로 기록했다 (원인이 안 남는다).
    """
    import re
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_policy import DEFAULT_PERCEPTION_CONFIG
    assert DEFAULT_PERCEPTION_CONFIG.is_absolute(), DEFAULT_PERCEPTION_CONFIG
    assert DEFAULT_PERCEPTION_CONFIG.is_file(), DEFAULT_PERCEPTION_CONFIG
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    arg = re.search(r'add_argument\("--perception-config".*?\)', src, re.S).group(0)
    assert "DEFAULT_PERCEPTION_CONFIG" in arg, arg
