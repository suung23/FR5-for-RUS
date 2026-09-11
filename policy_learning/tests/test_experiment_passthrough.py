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
