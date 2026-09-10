"""scripts/eval_policy.py 연기 시험 — 6 자유도 요약이 전부 나오는지."""

import json
import subprocess
import sys
from pathlib import Path

from conftest import small_config


def test_eval_policy_script_reports_all_six_axes(built_dataset, tmp_path):
    from rus_policy.train import Trainer

    cfg = small_config()
    cfg.train.epochs = 1
    tr = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run")
    tr.fit()
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "eval_policy.py"), str(tmp_path / "run" / "last.pt"),
         "--dataset", str(built_dataset), "--split", "val", "--device", "cpu",
         "--csv", str(tmp_path / "e.csv")],
        capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr[-3000:]
    js = json.loads(out.stdout[out.stdout.index("{"):out.stdout.rindex("}") + 1])
    for nm, un in (("x", "mm"), ("z", "mm"), ("thx", "deg"), ("thy", "deg"), ("thz", "deg")):
        assert f"mae_{nm}_{un}" in js, nm
    assert "nmae" in js
    assert (tmp_path / "e.csv").is_file()


def test_diag_quality_script_runs(built_dataset, tmp_path):
    from rus_policy.train import Trainer

    cfg = small_config()
    cfg.train.epochs = 1
    Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run").fit()
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "diag_quality.py"), str(tmp_path / "run" / "last.pt"),
         "--dataset", str(built_dataset), "--split", "val", "--device", "cpu"],
        capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "순위상관" in out.stdout and "민감도" in out.stdout
