#!/usr/bin/env bash
# 부호 수정 데이터셋 위의 학습 스윕. 목적은 하나 — **면내 방향**을 잃지 않는 모델.
#
# 기준선(A0, beta_adapt 기본)에서 관찰된 것: β 가 epoch 11 에 상한 32 에 붙고, 그때부터
# select_corr_dx_thy 가 −0.90 → −0.25 로 무너진다. z 가 붕괴하면 후보가 서로 같아지고,
# 그러면 실행시 선택이 관성 보너스에 먹힌다 (decided_frac 0 %). 그래서 β 를 묶는다.
#
# 체크포인트 기준을 select_nmae 에서 **select_corr_dx_thy** 로 바꾼다. 이 값이 실험의
# 사전 등록 평가지표(centroid_dx 와 지령의 상관)와 같은 양이고, 낮을수록(= 더 음수) 좋다.
set -u
cd "$(dirname "$0")/.."
source ../Unet_seg/.venv/bin/activate
M=select_corr_dx_thy
run () {   # 이름 그리고 --set 인자들
  local name="$1"; shift
  if [ -d "runs/$name" ]; then echo "== $name 이미 있음, 건너뜀"; return; fi
  echo "== $name  $(date +%H:%M:%S)"
  python3 scripts/train_policy.py --out "runs/$name" \
      --set train.num_workers=8 --set train.checkpoint_metric=$M "$@" \
      > "runs/$name.log" 2>&1
  echo "   끝  $(date +%H:%M:%S)  $(grep -c 'epoch ' runs/$name.log 2>/dev/null) 에폭"
}
run beta_fixed05  --set train.beta_adapt=false --set loss.beta_kl=0.5
run beta_fixed01  --set train.beta_adapt=false --set loss.beta_kl=0.1
run beta_cap2     --set train.beta_adapt=true  --set train.beta_max=2.0
echo "스윕 완료 $(date +%H:%M:%S)"
