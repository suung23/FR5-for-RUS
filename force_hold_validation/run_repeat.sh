#!/usr/bin/env bash
#
# 반복 한 회를 통째로 돌린다 — 접촉 한 번, 목표만 바꾸며 전 대역.
#
#   ./run_repeat.sh 1                    # 1 회차
#   ./run_repeat.sh 2 --hold 60          # 유지 구간을 60 s 로
#   ./run_repeat.sh 3 --targets "1.0 3.0"
#   ./run_repeat.sh 4 --step-ml 150
#
# 조작자가 할 일은 **주사기와 키 세 개뿐**이다. 목표를 바꾸고 힘이 앉기를 기다리는
# 것은 이 스크립트가 한다 — 고정 시간을 자는 대신 실제 힘이 밴드 안에 연속으로
# 머무는 것을 보고 넘어가므로, 팬텀이 무르든 단단하든 같은 절차가 성립한다.
#
# 프로브를 떼지 마라. 이탈 문턱 아래로 내려가면 접근 모드로 돌아가고, 그 자리에서
# 회차가 멈춘다 (스크립트가 그 사실을 알려 준다).
set -uo pipefail

NODE=/us_diff_ik_node
TARGETS="0.5 1.0 1.5 2.0 3.0 4.0"
HOLD_S=30
STEP_ML=500
# 캡처를 담을 디렉터리. 실험 하나를 통째로 새 디렉터리에 담으면, 분석이 예전
# 실행과 섞이지 않는다 — run_force_hold_analysis.py 는 디렉터리 전체를 읽는다.
OUT_DIR=runs

REPEAT="${1:-}"
[[ -z "$REPEAT" ]] && {
  echo "사용법: $0 <회차> [--hold N] [--targets \"...\"] [--step-ml N] [--out-dir DIR]" >&2; exit 2; }
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hold)    HOLD_S="$2"; shift 2 ;;
    --targets) TARGETS="$2"; shift 2 ;;
    --step-ml) STEP_ML="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done

command -v ros2 >/dev/null || { echo "ros2 가 없다. setup.bash 를 source 하라." >&2; exit 1; }
ros2 node list 2>/dev/null | grep -qx "$NODE" || {
  echo "$NODE 가 안 보인다. start_session.sh 가 떠 있는가?" >&2; exit 1; }

label() { printf 't%s_r%s' "$(echo "$1" | tr . p)" "$2"; }
getp() { ros2 param get "$NODE" "$1" 2>/dev/null | awk '{print $NF}'; }

# ── 전제조건을 스크립트가 직접 건다 ──────────────────────────────────────
# 이걸 손으로 거는 준비 단계로 두었더니 건너뛰기 쉬웠고, 건너뛰면 진입 문턱이
# 배포값(2.0 N)인 채로 돌아 **접촉을 해도 힘 제어가 켜지지 않는다.** 목표보다
# 문턱이 높으면 그 대역은 원리적으로 도달할 수 없다.
# 진입 문턱은 **목표에서 0.1 N 낮은 자리**다. 고정 저문턱(0.2)은 무접촉 잔차보다
# 낮아 자유공간에서 접촉으로 오인됐다 — 그러면 팔이 목표 힘을 찾아 스스로 전진하며,
# 앞에 무엇이 있든 그렇게 한다. 목표에 붙여 두면 문턱이 항상 실제 접촉 수준이고,
# 붙는 순간 조절기가 남은 0.1 N 을 스스로 좁힌다.
ENTER_OFFSET_N=0.1
RELEASE_N=0.10    # 이 아래로 0.5 s 지속되면 접촉이 끝난 것으로 본다.
BAND_N=0.05       # 수렴 허용오차 — 센서 잡음 바닥.

entry_for() { awk -v t="$1" -v d="$ENTER_OFFSET_N" 'BEGIN{ printf "%.2f", t - d }'; }

echo "전제조건을 건다 (진입 = 목표 − $ENTER_OFFSET_N · 이탈 $RELEASE_N · 밴드 $BAND_N)"
for kv in "teleop.contact_probing_release_n:$RELEASE_N" \
          "contact_control.deadband_n:$BAND_N"; do
  ros2 param set "$NODE" "${kv%%:*}" "${kv##*:}" >/dev/null || {
    echo "  ✗ ${kv%%:*} 설정 실패" >&2; exit 1; }
done
for tgt in $TARGETS; do
  e=$(entry_for "$tgt")
  awk -v e="$e" -v r="$RELEASE_N" -v b="$BAND_N" -v t="$tgt" 'BEGIN{
    if (r+0 >= e+0) { printf "  ✗ 목표 %s: 이탈 %s 이 진입 %s 이상이다\n", t, r, e; exit 1 }
    if (t-b <= r+0) { printf "  ✗ 목표 %s: 밴드 아래끝이 이탈 문턱 이하다\n", t; exit 1 }
  }' || exit 1
done
echo "  ✓ 이탈 $(getp teleop.contact_probing_release_n) · 밴드 $(getp contact_control.deadband_n)"
echo "  ✓ 진입: $(for t in $TARGETS; do printf '%s→%s ' "$t" "$(entry_for "$t")"; done)"
echo

total=$(echo "$TARGETS" | wc -w)
echo "── 회차 $REPEAT · 목표 $total 개 [$TARGETS] · 유지 ${HOLD_S}s · 스텝 ${STEP_ML} mL ──"
echo "   주사기 조작만 하면 된다.  i 주입 시작 · w 회수 시작 · q 저장 후 다음"
echo
n=0
for tgt in $TARGETS; do
  n=$((n + 1))
  echo "▶ [$n/$total] 목표 $tgt N"
  # 진입 문턱을 먼저 내려야 한다. 목표를 올린 뒤 문턱이 옛 목표에 남아 있으면,
  # 접촉이 끊겼다 다시 붙을 때 엉뚱한 힘에서 잡힌다.
  entry=$(entry_for "$tgt")
  ros2 param set "$NODE" teleop.contact_probing_force_n "$entry" >/dev/null || {
    echo "  진입 문턱 설정 실패 — 중단" >&2; exit 1; }
  ros2 param set "$NODE" contact_control.target_force_n "$tgt" >/dev/null || {
    echo "  param set 실패 — 중단" >&2; exit 1; }
  echo "  진입 $entry N · 목표 $tgt N"

  # 첫 목표에서는 접촉을 기다린다 — 조작자가 스크립트를 띄운 뒤 프로브를 눌러도
  # 되도록. 두 번째부터는 이미 접촉 중이므로 끊기면 그것이 사고다.
  if [[ $n -eq 1 ]]; then
    python3 wait_settled.py --target "$tgt" --timeout-s 60 --await-contact
  else
    python3 wait_settled.py --target "$tgt" --timeout-s 60
  fi
  case $? in
    0) ;;
    2) echo "  접촉이 끊겼다. 다시 접촉시킨 뒤 아래로 이어서 돌려라:" >&2
       echo "    ./run_repeat.sh $REPEAT --hold $HOLD_S --step-ml $STEP_ML --targets \"$(echo "$TARGETS" | tr ' ' '\n' | sed -n "/^$tgt\$/,\$p" | tr '\n' ' ')\"" >&2
       exit 3 ;;
    *) echo "  힘이 앉지 않았다 — 중단" >&2; exit 4 ;;
  esac

  python3 capture_force_hold.py --label "A_$(label "$tgt" "$REPEAT")" \
      --run-type hold --seconds "$HOLD_S" --out-dir "$OUT_DIR" || exit 1
  echo "  ── 교란: ${STEP_ML} mL 주입 3 회 · 회수 3 회. 시작할 때 i / w, 끝나면 q ──"
  python3 capture_force_hold.py --label "B_$(label "$tgt" "$REPEAT")" \
      --run-type disturbance --step-ml "$STEP_ML" --out-dir "$OUT_DIR" || exit 1
  echo
done
echo "회차 $REPEAT 끝. 프로브를 떼고, 접촉 지점과 각도를 바꿔 다음 회차로."
