#!/usr/bin/env python3
"""논문용 회전 QC 표 — QC 결과 JSON 에서 LaTeX 을 찍는다.

    python3 report/make_table_rotation.py                    # 화면에 LaTeX
    python3 report/make_table_rotation.py --out report/tab_rotation.tex
    python3 report/make_table_rotation.py --format markdown  # 초고 확인용

그림 스크립트와 같은 규약이다 — **숫자는 전부 JSON 에서 읽고 손으로 적은 값은
없다.** 논문 표를 손으로 옮기면 재분석 후 본문만 옛 숫자로 남는다.

표 두 개를 낸다:

  1. 채널별 추적 정확도 (주 소스 host6). 논문에 반드시 들어가는 표.
  2. 6 축 대 9 축 비교. 자력계를 뺀 쪽이 나은 것이 이 리그의 결과라, 그 주장을
     받치려면 같이 실어야 한다.

부호 규약은 캡션에 박는다. 실효 지연이 음수면 IMU 가 로봇 상태 스트림보다
**앞선다**는 뜻이고 (``xcorr_lag`` docstring), 이는 로봇 UDP 상태 패킷의 전송
지연이 IMU 자신의 지연보다 크기 때문이다. 즉 이 값은 IMU 의 절대 지연이 아니라
두 스트림의 상대 지연이다 — 캡션에서 그렇게 읽히지 않으면 오해를 부른다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_QC = os.path.dirname(_HERE)

#: JSON 채널 이름 → 논문 표기. `spin` 은 프로브 장축 둘레 회전이라 축회전으로 쓴다.
CHANNELS = (
    ("tip_x", r"Tilt $x$"),
    ("tip_y", r"Tilt $y$"),
    ("spin", "Axial rotation"),
    ("tilt", "Combined tilt"),
)

#: 소스 이름 → 논문 표기.
SOURCE_LABEL = {
    "host6": "6-axis (host fusion, no magnetometer)",
    "host9": "9-axis (host fusion)",
    "chip": "9-axis (on-chip rotation vector)",
}


def load(path: str) -> dict:
    """QC 결과 JSON 을 읽는다."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(value, digits: int, plus: bool = False) -> str:
    """유효숫자를 고정해 찍는다. 값이 없으면 표에서 흔히 쓰는 대시로."""
    if value is None or value != value:            # None 또는 NaN
        return "--"
    sign = "+" if plus else ""
    return f"{value:{sign}.{digits}f}"


def channel_rows(rot: dict) -> list[tuple]:
    """채널 네 개 + geodesic 한 줄. 표에 들어갈 순서 그대로."""
    ch = rot["channels"]
    rows = []
    for key, label in CHANNELS:
        c = ch[key]
        rows.append((
            label,
            _fmt(c["amp_gt_deg"], 1),
            _fmt(c["latency_ms"], 1, plus=True),
            _fmt(c["corr"], 4),
            _fmt(c["err_rms_deg"], 2),
            _fmt(c["err_rms_delay_corrected_deg"], 2),
            _fmt(c["err_p95_deg"], 2),
            _fmt(c["drift_deg_per_min"], 2, plus=True),
        ))
    g = rot["geodesic"]
    rows.append((
        r"3D geodesic",
        "--",
        _fmt(g["lag_ms"], 1, plus=True),
        "--",
        _fmt(g["rms_deg"], 2),
        _fmt(g["rms_delay_corrected_deg"], 2),
        _fmt(g["p95_deg"], 2),
        "--",
    ))
    return rows


def compare_rows(res: dict, a: str, b: str) -> list[tuple]:
    """두 소스의 같은 지표를 나란히. 마지막 열은 배수다."""
    ra, rb = res["rotation"][a], res["rotation"][b]

    def ratio(x, y):
        return "--" if not y else _fmt(x / y, 1) + r"$\times$"

    out = []
    for label, va, vb in (
        (r"Geodesic RMS [\si{\degree}]",
         ra["geodesic"]["rms_deg"], rb["geodesic"]["rms_deg"]),
        (r"Geodesic 95th pct.\ [\si{\degree}]",
         ra["geodesic"]["p95_deg"], rb["geodesic"]["p95_deg"]),
        (r"Axial rotation RMS [\si{\degree}]",
         ra["channels"]["spin"]["err_rms_deg"], rb["channels"]["spin"]["err_rms_deg"]),
        (r"Combined tilt RMS [\si{\degree}]",
         ra["channels"]["tilt"]["err_rms_deg"], rb["channels"]["tilt"]["err_rms_deg"]),
        (r"Axial drift [\si{\degree\per\minute}]",
         ra["channels"]["spin"]["drift_deg_per_min"],
         rb["channels"]["spin"]["drift_deg_per_min"]),
    ):
        digits = 2 if "drift" not in label.lower() else 2
        out.append((label, _fmt(va, digits, plus="drift" in label.lower()),
                    _fmt(vb, digits, plus="drift" in label.lower()),
                    ratio(abs(vb), abs(va))))
    return out


def latex(res: dict, src: str) -> str:
    """booktabs + siunitx 표 두 개.

    숫자 열은 ``S`` 로 잡는다. 소수점이 맞춰지고 음수가 하이픈이 아니라 진짜
    빼기 기호로 나온다 — 표에서 ``-22.8`` 과 ``$-$22.8`` 은 다르게 보인다.
    ``S`` 열에서 글자를 쓰려면 중괄호로 싸야 하므로 머리글과 대시가 모두 감싸져
    있다.
    """
    rot = res["rotation"][src]
    other = "chip" if src != "chip" else "host6"
    g = rot["geodesic"]
    tb = res["timebase"][rot["block"]]
    al = res["alignment"]

    # 열 서식. + 는 부호 자리를 늘 비워 두어 부호 유무로 열이 흔들리지 않게 한다.
    cols = ("l "
            "S[table-format=2.1] "
            "S[table-format=+2.1] "
            "S[table-format=1.4] "
            "S[table-format=2.2] "
            "S[table-format=2.2] "
            "S[table-format=2.2] "
            "S[table-format=+1.2]")
    head = " & ".join([
        "Channel", "{Ref.\\ amplitude}", "{Effective latency}", "{$\\rho$}",
        "{RMS error}", "{RMS (delay corr.)}", "{95th pct.}", "{Drift}",
    ])
    units = " & ".join([
        "", "{[\\si{\\degree}]}", "{[\\si{\\milli\\second}]}", "{}",
        "{[\\si{\\degree}]}", "{[\\si{\\degree}]}", "{[\\si{\\degree}]}",
        "{[\\si{\\degree\\per\\minute}]}",
    ])

    rows = channel_rows(rot)
    body = "\n".join("    " + " & ".join(r) + r" \\" for r in rows[:-1])
    geo = "    " + " & ".join("{--}" if c == "--" else c for c in rows[-1]) + r" \\"

    cmp_body = "\n".join(
        "    " + " & ".join(r) + r" \\" for r in compare_rows(res, src, other)
    )

    return rf"""% 자동 생성: report/make_table_rotation.py — 손으로 고치지 말 것.
% 필요 패키지: \usepackage{{booktabs}}, \usepackage{{siunitx}}
\begin{{table}}[t]
  \centering
  \caption{{Orientation tracking accuracy of the wrist-mounted IMU against robot
  forward kinematics, recorded during teleoperated ultrasound probing
  (\SI{{{rot['span_s']:.0f}}}{{\second}}, $n = {g['n']}$ paired samples;
  sensor \SI{{{tb['imu_hz']:.0f}}}{{\hertz}}, ground truth
  \SI{{{tb['gt_hz']:.0f}}}{{\hertz}}). Sensor source:
  {SOURCE_LABEL.get(src, src)}. Hand--eye alignment residual
  \SI{{{al['resid_rms_deg']:.2f}}}{{\degree}} RMS over $n = {al['n']}$ static
  poses. Tilt $x$ and tilt $y$ are the two components of the probe axis
  deviation from vertical; axial rotation is rotation about the probe axis.
  A negative effective latency means the IMU \emph{{leads}} the robot state
  stream. The quantity is the relative lag between the two streams, not the
  absolute latency of either, because the robot's UDP status packets carry
  their own transport delay.}}
  \label{{tab:imu-rotation}}
  \begin{{tabular}}{{{cols}}}
    \toprule
    {head} \\
    {units} \\
    \midrule
{body}
    \midrule
{geo}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Effect of including the magnetometer, evaluated on the same
  recording with the same hand--eye alignment. Omitting it improves every
  channel. The gain is largest in axial rotation, the axis whose observability
  depends on the magnetic reference: the ferromagnetic robot arm perturbs the
  local field as the pose changes.}}
  \label{{tab:imu-6ax-vs-9ax}}
  \begin{{tabular}}{{l S[table-format=+2.2] S[table-format=+2.2] r}}
    \toprule
    Metric & {{6-axis}} & {{9-axis}} & {{Ratio}} \\
    \midrule
{cmp_body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def markdown(res: dict, src: str) -> str:
    """초고에서 눈으로 확인할 용도. 논문에 넣는 것은 LaTeX 쪽이다."""
    rot = res["rotation"][src]
    other = "chip" if src != "chip" else "host6"
    head = ("| Channel | Ref. amp. [deg] | Latency [ms] | rho | RMS [deg] "
            "| RMS delay-corr. [deg] | p95 [deg] | Drift [deg/min] |")
    sep = "|" + "---|" * 8
    rows = ["| " + " | ".join(r).replace(r"$x$", "x").replace(r"$y$", "y") + " |"
            for r in channel_rows(rot)]
    cmp_head = "| Metric | 6-axis | 9-axis | Ratio |"
    cmp_rows = ["| " + " | ".join(r) + " |" for r in compare_rows(res, src, other)]
    clean = lambda s: (s.replace(r"\si{\degree\per\minute}", "deg/min")     # noqa: E731
                        .replace(r"\si{\degree}", "deg")
                        .replace(r"$\times$", "x").replace(r"\ ", " "))
    return "\n".join([head, sep, *rows, "", cmp_head, "|---|---|---|---|",
                      *[clean(r) for r in cmp_rows]])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="회전 QC 표를 논문 형식으로 낸다")
    p.add_argument("--run", default=os.path.join(_QC, "raw_data"),
                   help="세션 폴더 (기본 %(default)s)")
    p.add_argument("--source", default=None,
                   help="센서 소스. 기본은 JSON 의 primary_source")
    p.add_argument("--format", choices=("latex", "markdown"), default="latex")
    p.add_argument("--out", default=None, help="파일로 저장. 없으면 표준출력")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    src_hint = args.source or "host6"
    path = os.path.join(args.run, "meta", f"qc_{src_hint}.json")
    if not os.path.exists(path):
        path = os.path.join(args.run, "meta", "qc_result.json")
    res = load(path)
    src = args.source or res.get("primary_source", "host6")
    if src not in res.get("rotation", {}):
        p.error(f"소스 {src!r} 가 결과에 없다: {sorted(res.get('rotation', {}))}")

    text = latex(res, src) if args.format == "latex" else markdown(res, src)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        print(f"{args.out} 에 썼다 (source={src}, block={res['rotation'][src]['block']})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
