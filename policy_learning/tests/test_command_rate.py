"""정책의 발행 주기가 제어 스택의 워치독과 맞물리는가.

2026-09-11: 정책은 policy_hz 5 Hz(200 ms)로 냈고, 제어 스택의 워치독은 **50 Hz
발행자** 기준으로 맞춰져 있었다 (probe.yaml: twist_hold_s 0.04 = "발행 주기 20 ms 의
2배", twist_timeout_s 0.1). 그래서 매 200 ms 주기가 이렇게 됐다:

    0–40 ms    그대로
    40–100 ms  1.0 → 0 으로 선형 감쇠
    100–200 ms **두절** — 접촉 프로빙 분기가 zeros(6) 을 넣어 회전 세 축이 사라진다

평균 0.35 배에 매 주기 절반이 정확히 0. 로봇은 "거의 안 움직이는" 것처럼 보였고,
그 원인은 정책·제어 어느 쪽 코드에도 없었다 — 둘 사이의 **주기 불일치**였다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE_YAML = ROOT / "fr5_control" / "fr5_control" / "config" / "probe.yaml"
RUN_POLICY = ROOT / "policy_learning" / "scripts" / "run_policy.py"


def _yaml_number(name: str) -> float:
    m = re.search(rf"^\s*{re.escape(name)}:\s*([0-9.]+)", PROBE_YAML.read_text(), re.M)
    assert m, f"probe.yaml 에 {name} 이 없다"
    return float(m.group(1))


def _default_of(flag: str) -> float:
    """run_policy.py 의 add_argument 기본값."""
    src = RUN_POLICY.read_text()
    m = re.search(rf'add_argument\("{re.escape(flag)}",\s*type=float,\s*default=([0-9.]+)', src)
    assert m, f"{flag} 의 기본값을 못 찾았다"
    return float(m.group(1))


def test_republish_is_fast_enough_for_the_watchdog():
    timeout_s = _yaml_number("twist_timeout_s")
    hold_s = _yaml_number("twist_hold_s")
    hz = _default_of("--republish-hz")
    period_s = 1.0 / hz

    # 두절이 되기 전에 다음 발행이 와야 한다.
    assert period_s < timeout_s, (
        f"재발행 주기 {period_s * 1000:.0f} ms 가 워치독 {timeout_s * 1000:.0f} ms 보다 "
        "길다 — 매 주기 일부가 두절로 잡히고 그때 정책 회전이 0 이 된다")
    # 감쇠(_hold_fade)조차 시작되지 않아야 지령이 온전히 실린다.
    assert period_s <= hold_s, (
        f"재발행 주기 {period_s * 1000:.0f} ms 가 twist_hold {hold_s * 1000:.0f} ms 보다 "
        "길다 — 지령이 감쇠 구간에 걸려 요청한 속도보다 느리게 실린다")


def test_republish_is_on_by_default():
    """꺼져 있으면 policy_hz 만으로 내게 되고, 그것이 2026-09-11 의 상태였다."""
    assert _default_of("--republish-hz") > 0.0


def test_held_command_expires_so_the_watchdog_still_works():
    """재발행은 추론을 **대신하지 않는다.**

    _tick 이 멎어도 재발행 타이머는 살아 있다. 나이 제한이 없으면 마지막 지령이
    영원히 나가 워치독이 영영 안 걸린다 — 아무도 판단하지 않는 속도로 로봇이 돈다.
    """
    src = RUN_POLICY.read_text()
    assert "_held_max_age_s" in src, "재발행에 나이 제한이 없다"
    assert re.search(r"_held_max_age_s\s*=\s*3\.0\s*/\s*float\(cfg\.timing\.policy_hz\)", src), \
        "나이 제한이 추론 주기에 묶여 있어야 한다 (세 주기)"
    assert re.search(r"if \(time\.time\(\) - self\._held_t\) > self\._held_max_age_s", src), \
        "_republish 가 나이를 검사하지 않는다"


def test_idle_stops_republishing():
    """정책이 판단을 멈추면 지령도 멈춰야 한다 — _idle 은 _stop 을 부르고 _stop 은 비운다."""
    src = RUN_POLICY.read_text()
    stop = src[src.index("    def _stop(self):"):src.index("    def _republish(self):")]
    assert "self._held = None" in stop, "_stop 이 보유 지령을 비우지 않는다"
    idle = src[src.index("    def _idle(self, reason: str)"):]
    idle = idle[:idle.index("    def _tick(self):")]
    assert "self._stop()" in idle, "_idle 이 _stop 을 부르지 않는다"


def test_log_shows_policy_intent_separately_from_command():
    """hold 에피소드에서 '정책이 0 을 낸다' 로 읽히지 않아야 한다.

    2026-09-11: 로그의 ω 는 조건을 지난 **지령**만 찍었다. hold 는 지령을 0 으로 막으므로
    한 에피소드 내내 ω=(+0.00,+0.00,+0.00) 이 찍혔고, 조작자는 모델이 아무것도 안 낸다고
    읽었다. 실제 net_thx 는 평균 −4.53°/chunk 로 컸다.

    정책의 의도는 **위약 무작위화보다 먼저** 떠 두어야 한다. 뒤에서 뜨면 위약 에피소드의
    '정책' 줄이 무작위 방향을 정책 의도라고 말하게 된다.
    """
    src = RUN_POLICY.read_text()
    i_copy = src.index("a_policy = a.copy()")
    i_placebo = src.index("a = randomize_direction(a, self.rng)")
    assert i_copy < i_placebo, "a_policy 를 위약 무작위화 뒤에 떴다 — 위약의 의도가 섞인다"

    log = src[src.index("if len(self.rows) % 10 == 0:"):]
    log = log[:log.index("def _judge_now")]
    assert "정책  ω=" in log and "지령  ω=" in log, "로그가 정책 의도와 지령을 나눠 찍지 않는다"
    assert "a_policy[AX_ANG]" in log, "정책 줄이 a_policy 를 읽지 않는다"
