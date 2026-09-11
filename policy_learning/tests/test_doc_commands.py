"""정적 검사 — 런북의 명령이 **실행 위치에 의존하는가**.

문서의 명령은 붙여넣기용이다. 붙여넣는 사람은 자기가 어느 디렉터리에 있는지
문서와 맞춰 주지 않는다. 2026-09-11 에 두 번 깨졌다:

    ~/FR5-for-RUS/policy_learning$ python3 policy_learning/scripts/analyze_experiment.py …
    can't open file '…/policy_learning/policy_learning/scripts/analyze_experiment.py'

규칙: ```bash 블록 안에서 저장소 상대경로를 쓰려면 **그 블록 안 앞쪽에 `cd` 가 있어야**
한다. 없으면 `~/FR5-for-RUS/...` 절대경로로 쓴다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["docs/RUNBOOK_REAL_RUN.md", "docs/PROTOCOL_EXPERIMENT.md"]

# 저장소 루트나 policy_learning 기준이라야 도는 경로들.
REPO_RELATIVE = re.compile(
    r"(?:^|\s)(?:\./scripts/|\./|"
    r"python3\s+(?:scripts/|policy_learning/|phantom_stiffness/|fr5_control/))"
)


def _offenders(text: str) -> list[str]:
    bad, in_block, saw_cd = [], False, False
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            if in_block:
                in_block, saw_cd = False, False
            else:
                in_block = line.strip().lower() in ("```bash", "```sh", "```")
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^cd\s", stripped):
            saw_cd = True
            continue
        if not saw_cd and REPO_RELATIVE.search(line):
            bad.append(f"{i}: {stripped}")
    return bad


def test_doc_commands_do_not_depend_on_cwd():
    offenders = []
    for rel in DOCS:
        for hit in _offenders((ROOT / rel).read_text()):
            offenders.append(f"{rel}:{hit}")
    assert not offenders, (
        "블록 안에 `cd` 없이 저장소 상대경로를 쓴다 — 붙여넣는 위치에 따라 깨진다.\n  "
        + "\n  ".join(offenders)
        + "\n  `~/FR5-for-RUS/...` 로 쓰거나 블록 맨 앞에 `cd` 를 넣어라."
    )


def test_detector_catches_the_2026_09_11_regression():
    """검사기가 실제로 도는지. 안 돌면 위 테스트는 늘 통과한다."""
    broken = "```bash\npython3 policy_learning/scripts/analyze_experiment.py x\n```\n"
    with_cd = "```bash\ncd ~/FR5-for-RUS\npython3 policy_learning/scripts/analyze_experiment.py x\n```\n"
    absolute = "```bash\npython3 ~/FR5-for-RUS/policy_learning/scripts/analyze_experiment.py x\n```\n"
    assert len(_offenders(broken)) == 1
    assert _offenders(with_cd) == []
    assert _offenders(absolute) == []
