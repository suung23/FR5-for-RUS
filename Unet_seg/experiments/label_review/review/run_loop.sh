#!/usr/bin/env bash
# 판정 → 필터 → 학습 → 평가 → 비교. review/verdicts.csv 를 읽어 한 번에 돌린다.
#
#   ./run_loop.sh                      전 필터, 전 split 일관 적용 (권장)
#   ./run_loop.sh --protect-test       test 는 건드리지 않는다 (기존 런과 비교용)
#   ./run_loop.sh --no-hydro           hydro 면적 필터 없이
#   NAME=myrun EPOCHS=40 ./run_loop.sh
#
# 기본값은 label_wrong 만 제외한다. unjudgeable 은 "틀렸다"가 아니라 "사람이 판단할
# 근거가 없다"는 뜻이라 자동으로 버리지 않는다. 둘 다 버리려면:
#   DROP=label_wrong,unjudgeable ./run_loop.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."          # -> Unet_seg
PY=.venv/bin/python
DATA=/home/rosotauser/datasets/pfus
REVIEW=experiments/label_review/review
NAME=${NAME:-reviewed}
DROP=${DROP:-label_wrong}
EPOCHS=${EPOCHS:-40}
HYDRO=0.022888
PROTECT=""

for arg in "$@"; do
  case "$arg" in
    --protect-test) PROTECT="--protect-splits test" ;;
    --no-hydro)     HYDRO=0 ;;
    *) echo "알 수 없는 인자: $arg"; exit 2 ;;
  esac
done

MANIFEST=$DATA/manifest_$NAME.csv
CKPT=checkpoints/pfus_bladder_$NAME
CFG=configs/pfus_bladder_$NAME.yaml

# ---------------------------------------------------------------- 1. 제외 명단
EXCLUDE=$(DROP="$DROP" $PY - <<'EOF'
import csv, os
p = "experiments/label_review/review/verdicts.csv"
if not os.path.exists(p):
    raise SystemExit("verdicts.csv 가 없습니다. 먼저 server.py 로 판정하십시오.")
drop = set(os.environ["DROP"].split(","))
bad = sorted(r["patient_id"] for r in csv.DictReader(open(p)) if r["verdict"] in drop)
if not bad:
    raise SystemExit("제외할 환자가 없습니다.")
print(",".join(bad))
EOF
)
echo "── 제외 판정: $DROP"
echo "── 제외 환자: $EXCLUDE"

# ---------------------------------------------------------------- 2. 매니페스트
$PY scripts/filter_manifest_by_area.py \
  --manifest $DATA/manifest_edited.csv \
  --output   "$MANIFEST" \
  --min-patient-area-ratio $HYDRO \
  --exclude-patients "$EXCLUDE" \
  $PROTECT \
  --report   "${MANIFEST%.csv}_report.json"

# ---------------------------------------------------------------- 3. 설정 파일
cat > "$CFG" <<EOF
# 자동 생성 — review/run_loop.sh. 손으로 고치면 다음 실행에 덮어써진다.
# 제외: $EXCLUDE
# hydro: $HYDRO   protect: ${PROTECT:-none}
defaults: pfus_bladder_edit.yaml
experiment:
  name: pfus_bladder_$NAME
  output_dir: runs/pfus_bladder_$NAME
  seed: 42
data:
  manifest: $MANIFEST
  root: $DATA
train:
  checkpoint_dir: $CKPT
EOF

# ---------------------------------------------------------------- 4. 학습
echo "── 학습 시작 ($EPOCHS epoch) — checkpoints: $CKPT"
$PY scripts/train.py --config "$CFG" --epochs "$EPOCHS" 2>&1 | tail -5

# ---------------------------------------------------------------- 5. 평가
for split in test val; do
  for ck in best last; do
    echo "── 평가 $ck.pt / $split"
    $PY scripts/evaluate.py --config "$CFG" --checkpoint "$CKPT/$ck.pt" \
      --split $split --output-dir "runs/$NAME/${ck}_${split}" >/dev/null 2>&1
  done
done

# ---------------------------------------------------------------- 6. 비교
$PY - "$NAME" <<'EOF'
import json, os, sys, statistics as st
name = sys.argv[1]
prev = {"edit_man_hydro best/test": "runs/fullfilter/hydro_best_test/evaluation.json",
        "edit_man_hydro best/val":  "runs/fullfilter/hydro_best_val/evaluation.json"}
rows = []
for ck in ("best", "last"):
    for sp in ("test", "val"):
        p = f"runs/{name}/{ck}_{sp}/evaluation.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); o = d["spatial"]["overall"]; pp = d["spatial"]["per_patient"]
        rows.append((f"{name} {ck}/{sp}", d["num_patients"], d["num_frames"],
                     st.mean(v["dice"]["mean"] for v in pp.values()),
                     o["dice"]["mean"], o["iou"]["mean"]))
for label, p in prev.items():
    if os.path.exists(p):
        d = json.load(open(p)); o = d["spatial"]["overall"]; pp = d["spatial"]["per_patient"]
        rows.append((label, d["num_patients"], d["num_frames"],
                     st.mean(v["dice"]["mean"] for v in pp.values()),
                     o["dice"]["mean"], o["iou"]["mean"]))
print(f"\n{'런':28}{'환자':>5}{'프레임':>8}{'Dice(환자)':>12}{'Dice(프레임)':>13}{'IoU':>8}")
for r in rows:
    print(f"{r[0]:28}{r[1]:5d}{r[2]:8d}{r[3]:12.4f}{r[4]:13.4f}{r[5]:8.4f}")
print("\n※ 코호트가 다르면 Dice 를 직접 비교하지 마십시오. 환자 수를 먼저 보십시오.")
EOF
