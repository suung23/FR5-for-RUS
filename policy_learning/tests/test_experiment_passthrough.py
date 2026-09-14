"""실험 드라이버가 러너 옵션을 넘기는가, 그리고 방향 관성(gamma)을 바꿀 수 있는가.

2026-09-11: run_experiment 는 parse_args 와 위치 인자 extra 를 썼다. 선언 안 된 옵션은
거부됐고, `--` 를 붙여도 extra 가 앞에서 이미 빈 값으로 소비돼 역시 거부됐다 — 러너 전용
옵션을 드라이버 너머로 넘기는 통로가 사실상 없었다.
"""

import argparse
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _parser(name: str) -> argparse.ArgumentParser:
    src = (SCRIPTS / name).read_text()
    body = src[src.index("def main()"):]
    end = ("args, unknown = p.parse_known_args()" if "parse_known_args" in body
           else "args = p.parse_args")
    body = body[:body.index(end)]
    lines = [ln[4:] for ln in body.splitlines()[1:] if ln.startswith("    ")]
    ns = {"argparse": argparse}
    exec("\n".join(lines), ns)
    return ns["p"]


def _forwarded(argv):
    a, unknown = _parser("run_experiment.py").parse_known_args(argv)
    return a, list(a.extra) + [u for u in unknown if u != "--"]


def test_undeclared_runner_options_reach_the_runner():
    _, extra = _forwarded(["ck", "--out", "x", "--republish-hz", "40"])
    assert extra == ["--republish-hz", "40"]


def test_gamma_is_a_first_class_experiment_option():
    a, _ = _forwarded(["ck", "--out", "x", "--gamma", "0.005"])
    assert a.gamma == 0.005
    src = (SCRIPTS / "run_experiment.py").read_text()
    assert 'cmd += ["--gamma", str(args.gamma)]' in src


def test_gamma_defaults_to_the_checkpoint_value():
    """기본을 몰래 바꾸지 않는다 — 방향 관성은 실험 방법이므로 조작자가 고른다."""
    a, _ = _forwarded(["ck", "--out", "x"])
    assert a.gamma is None
    src = (SCRIPTS / "run_policy.py").read_text()
    assert "else float(cfg.train.gamma_mode_consistency)" in src
    assert '"gamma_mode_consistency": self.gamma' in src, "쓴 gamma 가 meta 에 안 남는다"


def test_blind_reaches_the_runner_and_hides_the_condition():
    """위약 대조의 전제: 조작자가 조건을 모른다.

    2026-09-11: run_experiment 는 자기 출력만 가렸고 러너에 --blind 를 넘기지 않았다. 러너는
    "에피소드 시작 (조건 policy …)" 과 매 로그 줄의 [placebo — 방향 무작위] 로 조건을
    드러냈다. 위약에서는 정책 줄과 지령 줄의 방향이 달라 그 차이만으로도 드러났다.
    """
    exp = (SCRIPTS / "run_experiment.py").read_text()
    assert 'cmd.append("--blind")' in exp, "드라이버가 러너에 --blind 를 넘기지 않는다"

    run = (SCRIPTS / "run_policy.py").read_text()
    assert "(조건 {self._shown_condition()}" in run, "에피소드 시작 줄이 조건을 그대로 찍는다"
    blind_branch = run[run.index("if self.args.blind:\n                # **정책 줄을"):]
    blind_branch = blind_branch[:blind_branch.index("else:")]
    assert "정책  ω=" not in blind_branch, "눈가림에서 정책 줄을 찍으면 위약이 드러난다"
    assert "_shown_condition()" in blind_branch


# --- 결과표의 열 (2026-09-14 밤) -----------------------------------------------


def test_a_new_column_migrates_the_rows_already_written(tmp_path):
    """열이 늘어나도 이미 쓴 줄이 헤더와 어긋나면 안 된다.

    예전 ``append_summary`` 는 파일이 새것일 때만 헤더를 쓰고, 그 뒤로는
    ``row.keys()`` 순서대로 값만 이어 붙였다. 그 상태에서 열을 하나 추가하면
    이어 돌리는 세션의 줄이 **옆 칸으로 밀린 채** 기록되는데 파일은 멀쩡해 보인다.
    유지 지표 세 열을 추가하면서 실제로 밟을 수 있었던 자리다 (--resume 은 정상 운용).
    """
    import csv
    from scripts.run_experiment import append_summary

    append_summary(tmp_path, {"index": 1, "success": 1})
    append_summary(tmp_path, {"index": 2, "success": 0, "hold_q_mean": 0.61})

    rows = list(csv.DictReader(open(tmp_path / "summary.csv", encoding="utf-8")))
    assert [r["index"] for r in rows] == ["1", "2"]
    assert rows[0]["success"] == "1" and rows[1]["success"] == "0"
    # 옛 줄은 새 열이 빈 칸이어야지, 값이 밀려 들어오면 안 된다.
    assert rows[0]["hold_q_mean"] == ""
    assert rows[1]["hold_q_mean"] == "0.61"


def test_the_hold_quality_columns_are_reported(tmp_path):
    """합격선만으로는 결과표가 전부 0 이 된다 — 연속값과 조건 기반 지표를 함께 싣는다."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py").read_text("utf-8")
    for col in ("hold_q_mean", "hold_q_pass_fraction", "hold_good_fraction"):
        assert f'"{col}": verdict.get(' in src, col
