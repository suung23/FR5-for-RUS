"""정적 검사 — import 누락 같은 '실행해야만 터지는' 오류를 미리 잡는다.

import 스모크 테스트는 모듈이 로드되는지만 본다. `main()` 안에서만 부르는 함수의
import 가 빠져 있으면 로드는 성공하고 **실행할 때 NameError 로 터진다.** 실제로
그렇게 두 번 놓쳤다. pyflakes 는 undefined name 을 정적으로 잡으므로 그 구멍을 막는다.

스타일은 보지 않는다 (§FATAL) — 이 폴더에는 다른 사람이 작업 중인 파일도 들어오므로,
무해한 지적으로 남의 작업을 막지 않는다.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)

# 이 테스트의 목적은 **실행해야만 터지는 오류**를 잡는 것이지 스타일 단속이 아니다.
# 그래서 pyflakes 지적 중 아래 부류만 실패로 본다. 미사용 변수/import 같은 나머지는
# 무해하고, 남이 작업 중인 파일에 대해 잡음만 만든다.
FATAL = (
    "undefined name",
    "syntax error",
    "may be undefined",             # from-star 등으로 정의가 불확실한 이름
    "redefinition of unused",       # 같은 이름 두 번 import/def — 대개 실수다
)


def sources():
    out = []
    for sub in ("host", "test", "qc_track"):
        d = os.path.join(ROOT, sub)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
    return out


def test_no_undefined_names():
    pytest.importorskip("pyflakes")
    r = subprocess.run([sys.executable, "-m", "pyflakes", *sources()],
                       capture_output=True, text=True)
    bad = [ln for ln in (r.stdout + r.stderr).splitlines()
           if any(f in ln.lower() for f in FATAL)]
    assert not bad, "pyflakes 치명 지적:\n" + "\n".join(bad)
