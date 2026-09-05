#!/usr/bin/env bash
#
# 반복 한 회를 통째로 돌린다 — 접촉 한 번, 목표만 바꾸며 전 대역.
#
#   ./run_repeat.sh 1                    # 1 회차
#   ./run_repeat.sh 2 --hold 60          # 유지 구간을 60 s 로
#   ./run_repeat.sh 3 --targets "1.0 3.0"
#   ./run_repeat.sh 4 --step-ml 150
#   ./run_repeat.sh 1h --hold-only            # 유지 구간만 (교란 없이)
#   ./run_repeat.sh 2 --placebo               # 대조군까지 (교란을 제어 없이 한 번 더)
#
# --placebo 는 대역마다 교란을 **두 번** 받는다: B 는 힘 유지를 켠 채, C 는 끈 채.
# C 에서 로봇은 접촉 자세 그대로 z 를 놓고 힘을 기록만 한다 — 같은 주사기 조작이
# 제어 없이 얼마의 힘을 만드는가, 그것이 B 를 해석할 기준선이다. 그 기준선이 없으면
# "힘이 목표 근처에 머물렀다" 가 제어 덕분인지 교란이 원래 그 정도였는지 알 수 없다.
# 지금까지는 팬텀 강성으로 외삽해 반사실을 짐작했다 (fh/analysis.regulation_evidence);
# C 는 그것을 직접 잰다.
#
# ⚠️ 안전층은 대조군에서도 살아 있다. 한계 힘을 넘으면 여전히 후퇴한다.
#
# --hold-only 는 대역마다 A(유지)만 받고 B(교란)를 건너뛴다. 한 회차의 유지가
# 못 쓰게 나왔을 때 주사기 없이 그 부분만 다시 받기 위한 것이다. 회차 이름을
# 바꿔서 부르면(위의 1h) 원래 회차의 파일이 남는다 — 캡처는 같은 이름을 덮어쓰지
# 않고 거부하므로, 잊어도 데이터가 사라지지는 않는다.
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
# 목표를 바꾼 뒤 힘이 앉기를 기다리는 상한 [s]. 밖으로 뺀 이유는 이것이 회차를
# 실제로 멈추는 값이기 때문이다 — 2026-09-04 의 2.5 N 이 그렇게 멈췄다. 루프가
# 한계주기를 돌면 조용한 창이 주기(약 8.5 s)마다 2~3 s 씩만 열리므로, 60 s 는
# 그 창을 예닐곱 번 볼 뿐이다. B_z 를 올려 루프를 늦췄다면 이 값도 함께 올린다.
# 🔁 2026-09-04: 60 → 120. B_z 배포값이 1000 → 3000 이 되어 접근이 3 배 느리다.
SETTLE_S=120
# 교란(B)을 건너뛰고 유지(A)만 받는다.
HOLD_ONLY=0
# 교란을 제어 없이 한 번 더 받는다 (C). 위약 대조군이다.
PLACEBO=0
# 캡처를 담을 디렉터리. 실험 하나를 통째로 새 디렉터리에 담으면, 분석이 예전
# 실행과 섞이지 않는다 — run_force_hold_analysis.py 는 디렉터리 전체를 읽는다.
OUT_DIR=runs

REPEAT="${1:-}"
[[ -z "$REPEAT" ]] && {
  { echo "사용법: $0 <회차> [--hold N] [--targets \"...\"] [--step-ml N]"
    echo "              [--out-dir DIR] [--settle-s N] [--hold-only] [--placebo]"; } >&2; exit 2; }
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hold)    HOLD_S="$2"; shift 2 ;;
    --targets) TARGETS="$2"; shift 2 ;;
    --step-ml) STEP_ML="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --settle-s) SETTLE_S="$2"; shift 2 ;;
    --hold-only) HOLD_ONLY=1; shift ;;
    --placebo)  PLACEBO=1; shift ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done

command -v ros2 >/dev/null || { echo "ros2 가 없다. setup.bash 를 source 하라." >&2; exit 1; }
ros2 node list 2>/dev/null | grep -qx "$NODE" || {
  echo "$NODE 가 안 보인다. start_session.sh 가 떠 있는가?" >&2; exit 1; }

label() { printf 't%s_r%s' "$(echo "$1" | tr . p)" "$2"; }
getp() { ros2 param get "$NODE" "$1" 2>/dev/null | awk '{print $NF}'; }

# 멈춘 목표부터 끝까지 다시 도는 명령. **인자를 전부 되풀이한다** — 예전에는
# --out-dir 을 빼고 찍어서, 그대로 붙여 넣으면 남은 대역이 기본 runs/ 로 가
# 한 실험이 두 디렉터리에 쪼개졌다. 분석은 디렉터리 단위로 읽으므로 그 회차는
# 반쪽만 분석된다.
# 힘 유지를 켜고 끈다. 되돌리는 것을 잊으면 다음 대역이 통째로 못 쓰게 되므로,
# 실패를 삼키지 않고 그 자리에서 멈춘다.
set_hold() {
  ros2 param set "$NODE" contact_control.force_hold_enabled "$1" >/dev/null || {
    echo "  힘 유지 $1 설정 실패 — 중단" >&2; return 1; }
  # `ros2 param get` 은 "Boolean value is: True" 를 찍는다 — 파이썬 표기라 첫
  # 글자가 대문자다. 넘겨준 "true" 와 그대로 비교하면 **성공했는데도 항상 실패**로
  # 읽히고, 회차가 대조군 직전에 멈춘다. 읽은 쪽을 소문자로 내려 맞춘다.
  local got
  got=$(getp contact_control.force_hold_enabled)
  [[ "${got,,}" == "${1,,}" ]] || {
    echo "  힘 유지가 $1 로 안 바뀌었다 (지금 $got) — 중단" >&2; return 1; }
  return 0
}

resume_hint() {
  printf '    ./run_repeat.sh %s --hold %s --step-ml %s --out-dir %s --settle-s %s%s --targets "%s"\n' \
    "$REPEAT" "$HOLD_S" "$STEP_ML" "$OUT_DIR" "$SETTLE_S" \
    "$( (( HOLD_ONLY )) && printf ' --hold-only' )" \
    "$(echo "$TARGETS" | tr ' ' '\n' | sed -n "/^$1\$/,\$p" | tr '\n' ' ' | sed 's/ *$//')"
}

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
# 힘 유지가 켜진 상태에서 시작한다. 앞 회차가 대조군 도중에 멈췄으면 꺼진 채로
# 남아 있고, 그러면 이번 회차 전체가 조용히 대조군이 된다.
set_hold true || exit 1
echo "  ✓ 이탈 $(getp teleop.contact_probing_release_n) · 밴드 $(getp contact_control.deadband_n)"
echo "  ✓ 진입: $(for t in $TARGETS; do printf '%s→%s ' "$t" "$(entry_for "$t")"; done)"
echo

# 받을 파일 이름이 이미 있는지 **먼저** 본다. 캡처가 스스로 거부하기는 하지만,
# 그때는 이미 프로브를 눌러 접촉을 잡고 힘이 앉기를 기다린 뒤다. 회차 이름 하나
# 때문에 그 자리에서 멈추면 접촉을 다시 잡아야 하고, 접촉점이 바뀌면 그 회차의
# 앞뒤 대역이 서로 다른 자리에서 측정된다.
clash=()
for tgt in $TARGETS; do
  for prefix in A B C; do
    (( HOLD_ONLY )) && [[ "$prefix" != A ]] && continue
    [[ "$prefix" == C ]] && ! (( PLACEBO )) && continue
    f="$OUT_DIR/${prefix}_$(label "$tgt" "$REPEAT")_samples.csv"
    [[ -e "$f" ]] && clash+=("$f")
  done
done
if (( ${#clash[@]} )); then
  echo "✗ 이미 있는 캡처 ${#clash[@]} 개 — 회차 이름을 바꿔라 (예: $0 ${REPEAT}h ...)" >&2
  printf '    %s\n' "${clash[@]}" >&2
  exit 1
fi

total=$(echo "$TARGETS" | wc -w)
if (( HOLD_ONLY )); then
  echo "── 회차 $REPEAT · 목표 $total 개 [$TARGETS] · 유지 ${HOLD_S}s · 교란 없음 ──"
  echo "   주사기는 쓰지 않는다. 프로브만 붙여 두면 스크립트가 끝까지 간다."
else
  echo "── 회차 $REPEAT · 목표 $total 개 [$TARGETS] · 유지 ${HOLD_S}s · 스텝 ${STEP_ML} mL ──"
  echo "   주사기 조작만 하면 된다.  i 주입 시작 · w 회수 시작 · q 저장 후 다음"
  (( PLACEBO )) && echo "   대조군 포함 — 대역마다 교란을 두 번 (B 제어 켬 · C 제어 끔)"
fi
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
    python3 wait_settled.py --target "$tgt" --timeout-s "$SETTLE_S" --await-contact
  else
    python3 wait_settled.py --target "$tgt" --timeout-s "$SETTLE_S"
  fi
  case $? in
    0) ;;
    2) echo "  접촉이 끊겼다. 다시 접촉시킨 뒤 이어서 돌려라:" >&2
       resume_hint "$tgt" >&2
       exit 3 ;;
    # 여기서도 이어가는 법을 찍는다. 2026-09-04 에 2.5 N 이 이 가지로 멈췄는데,
    # 아무것도 안 찍혀서 남은 대역을 손으로 다시 세어야 했다. 멈춘 자리를 아는
    # 것은 스크립트뿐이므로, 말해 줄 수 있는 것도 스크립트뿐이다.
    *) echo "  ${SETTLE_S} s 안에 힘이 앉지 않았다 — 중단" >&2
       echo "  루프가 목표 주위를 도는 중이면 감쇠를 올리고 대기도 늘려라:" >&2
       echo "    ros2 param set $NODE contact_control.admittance_b_z 4000.0" >&2
       resume_hint "$tgt" >&2
       exit 4 ;;
  esac

  python3 capture_force_hold.py --label "A_$(label "$tgt" "$REPEAT")" \
      --run-type hold --seconds "$HOLD_S" --out-dir "$OUT_DIR" || exit 1
  if (( HOLD_ONLY )); then
    echo "  (교란 생략 — --hold-only)"
  else
    echo "  ── 교란 B (제어 켬): ${STEP_ML} mL 주입 3 회 · 회수 3 회. i / w, 끝나면 q ──"
    python3 capture_force_hold.py --label "B_$(label "$tgt" "$REPEAT")" \
        --run-type disturbance --step-ml "$STEP_ML" --out-dir "$OUT_DIR" || exit 1

    if (( PLACEBO )); then
      # 대조군. 같은 접촉점·같은 주입량을 **제어 없이** 한 번 더 받는다.
      #
      # 순서를 B → C 로 둔 이유: C 가 끝나면 프로브가 목표에서 밀려나 있을 수
      # 있고, 그 상태에서 B 를 받으면 두 팔의 출발점이 달라진다. 제어를 켠 쪽을
      # 먼저 받으면 양쪽 모두 "목표에 앉은 자리" 에서 출발한다.
      set_hold false || exit 1
      echo "  ── 교란 C (제어 끔 · 대조군): 같은 조작을 그대로 반복한다 ──"
      echo "     로봇은 z 를 놓는다. 힘이 어디까지 가는지가 이 캡처의 전부다."
      python3 capture_force_hold.py --label "C_$(label "$tgt" "$REPEAT")" \
          --run-type disturbance --step-ml "$STEP_ML" --out-dir "$OUT_DIR"
      rc=$?
      # 캡처가 어떻게 끝났든 제어를 되돌린다. 꺼진 채로 다음 대역에 들어가면
      # 그 대역은 힘이 앉지 않고, 앉지 않은 이유가 파일에는 안 남는다.
      set_hold true || exit 1
      (( rc == 0 )) || exit 1
    fi
  fi
  echo
done
echo "회차 $REPEAT 끝. 프로브를 떼고, 접촉 지점과 각도를 바꿔 다음 회차로."
