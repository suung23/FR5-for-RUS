"""scripts/ 공용 부트스트랩 — 설치 없이 ``python scripts/x.py`` 가 되게 한다."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from rus_policy.config import PolicyConfig, load_config  # noqa: E402

DEFAULT_CONFIG = PKG_ROOT / "configs" / "policy_default.yaml"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 (기본: configs/policy_default.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="설정 덮어쓰기 key.path=value (여러 번 가능)")
    parser.add_argument("-v", "--verbose", action="store_true")


def config_from_args(args: argparse.Namespace) -> PolicyConfig:
    setup_logging(args.verbose)
    path = Path(args.config) if args.config else None
    if path is not None and not path.is_file():
        raise SystemExit(f"설정 파일이 없습니다: {path}")
    return load_config(path, args.overrides)
