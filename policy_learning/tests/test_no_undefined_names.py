"""정적 검사 — 미정의 이름(F821).

``run_policy.py`` 의 tick 루프처럼 **rclpy 가 있어야 도는 코드**는 pytest 로 못 잡는다.
2026-09-11 에 그 경로에서 두 번 죽었다: 어댑터 생성자 인자 하나를 빠뜨렸고, tick 안에서
정의한 적 없는 ``t`` 를 썼다. 둘 다 실기 앞에서 발견됐다 — 여기서 잡아야 하는 종류다.

pyflakes 는 임포트 없이 AST 만 본다. 그래서 ROS 가 없어도 돈다.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "rus_policy", ROOT / "scripts"]


def test_no_undefined_names():
    pyflakes = pytest.importorskip("pyflakes")          # 없으면 건너뛴다 (설치는 선택)
    assert pyflakes
    out = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, TARGETS)],
                         capture_output=True, text=True)
    bad = [ln for ln in out.stdout.splitlines()
           if "undefined name" in ln or "local variable" in ln]
    assert not bad, "미정의 이름:\n  " + "\n  ".join(bad)
