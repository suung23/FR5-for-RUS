"""정적 검사 — ``set -u`` 아래에서 ROS setup 을 source 하는 스크립트.

/opt/ros/jazzy/setup.bash 8 행은 ``AMENT_TRACE_SETUP_FILES`` 를 확인 없이 읽는다.
nounset 아래에서 source 하면 그 자리에서 셸이 죽는다. 출력을 /dev/null 로 보내고
있으면 **아무것도 찍지 않고 rc=1 로 끝난다** — 로그에 단서가 한 줄도 안 남는다.

2026-09-11 에 ``start_policy_eval.sh`` 가 그 상태였다. 런북이 가리키는 명령인데
쓰인 날부터 한 번도 뜬 적이 없었고, 조용해서 아무도 눈치채지 못했다.

pytest 로는 못 잡는다 (ROS 가 있어야 재현된다). 그래서 본문을 읽어서 본다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES_ROS = re.compile(r"^\s*(?:\.|source)\s+\S*(?:setup\.bash|env\.sh)")


def _nounset_at_source(text: str) -> list[int]:
    """nounset 이 켜진 채로 ROS setup 을 source 하는 줄 번호."""
    nounset, bad = False, []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # `set -uo pipefail` · `set -u` 는 켜고, `set +u` 는 끈다.
        for m in re.finditer(r"\bset\s+([-+])([a-zA-Z]*)", stripped):
            sign, flags = m.group(1), m.group(2)
            if "u" in flags:
                nounset = sign == "-"
        if nounset and SOURCES_ROS.match(line):
            bad.append(i)
    return bad


def test_ros_setup_not_sourced_under_nounset():
    offenders = []
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        for line_no in _nounset_at_source(path.read_text()):
            offenders.append(f"{path.relative_to(ROOT)}:{line_no}")
    assert not offenders, (
        "set -u 가 켜진 채로 ROS setup 을 source 한다 — 그 자리에서 조용히 죽는다.\n  "
        + "\n  ".join(offenders)
        + "\n  source 앞뒤를 `set +u` / `set -u` 로 감싸라."
    )


def test_detector_catches_the_2026_09_11_regression():
    """검사기 자체가 도는지. 이것이 없으면 위 테스트는 늘 통과한다."""
    broken = "set -uo pipefail\nsource \"$WORKSPACE/env.sh\" >/dev/null 2>&1\n"
    fixed = "set -uo pipefail\nset +u\nsource \"$WORKSPACE/env.sh\"\nset -u\n"
    assert _nounset_at_source(broken) == [2]
    assert _nounset_at_source(fixed) == []
