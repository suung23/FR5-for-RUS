#!/usr/bin/env bash
#
# 힘 목표를 한 대역씩 올리며 평형점을 만든다 — 강성 측정용.
#
#   ./phantom_stiffness/step_targets.sh                       # 0.5 1.0 1.5 2.0 3.0 4.0
#   ./phantom_stiffness/step_targets.sh 0.5 1.0 2.0 3.0
#   ./phantom_stiffness/step_targets.sh --down 4.0 3.0 2.0 1.0 0.5   # 제하 (이력용)
#
# **왜 깊이를 직접 안 주는가.** 접촉 프로빙 모드에서 빔축(z) 은 힘 조절기가 잡는다.
# 조작자가 z 를 밀어도 조절기가 목표 힘으로 되돌린다 — 즉 이 구성에서 조작자가
# 정할 수 있는 것은 **깊이가 아니라 힘**이다. 그래서 목표 힘을 바꿔 로봇이 앉는
# 자리(평형 깊이) 를 읽는다. `force_hold_validation` 이 0.3739 N/mm 를 얻은 것과
# 같은 경로다.
#
# 이 스크립트는 **파라미터만 바꾸고 기다린다.** 점을 찍는 것은 조작자다 (GUI 의 space).
# 힘 한계와 후퇴는 그대로 us_diff_ik_node 가 관리한다.
set -uo pipefail

NODE=${NODE:-/us_diff_ik_node}
NS=${NS:-/fr5_right}
ENTER_OFFSET_N=0.1      # 진입 문턱 = 목표 − 이 값
RELEASE_N=0.10          # 이 아래로 0.5 s 지속되면 접촉이 끝난 것으로 본다
BAND_N=0.05             # 수렴 허용오차 — 센서 잡음 바닥
SETTLE_S=${SETTLE_S:-120}
HERE="$(cd "$(dirname "$0")" && pwd)"
WAIT_SETTLED="$HERE/../force_hold_validation/wait_settled.py"

DIRECTION="load"
if [[ "${1:-}" == "--down" ]]; then DIRECTION="unload"; shift; fi
TARGETS=("$@")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(0.5 1.0 1.5 2.0 3.0 4.0)

# `ros2 param set` 은 **실패해도 종료 코드 0 을 준다** ("Setting parameter failed:" 를
# 찍고 성공으로 끝난다). 그래서 `|| exit` 가드가 통째로 무력했다 (2026-09-10 확인).
# 여기서는 **되읽어 값을 확인한다** — 목표 힘이 안 바뀐 채로 "앉았다" 로 넘어가면
# 로봇은 옛 목표를 쥐고 있는데 조작자는 바뀐 줄 안다. 접촉 중이면 그 차이가 힘이 된다.
setp() {
  local key="$1" want="$2" out got
  out=$(ros2 param set "$NODE" "$key" "$want" 2>&1) || true
  if ! grep -qi "successful" <<<"$out"; then
    echo "  ✗ $key 설정 실패: ${out//$'\n'/ }" >&2
    return 1
  fi
  got=$(ros2 param get "$NODE" "$key" 2>/dev/null | awk '{print $NF}')
  # 부동소수는 표기가 다를 수 있으므로 수치로 비교한다 (참/거짓은 문자열 비교).
  if [[ "$want" == "true" || "$want" == "false" ]]; then
    [[ "${got,,}" == "$want" ]] && return 0
  elif awk -v a="$got" -v b="$want" 'BEGIN{exit !(a+0==b+0)}' 2>/dev/null; then
    return 0
  fi
  echo "  ✗ $key 되읽기 불일치: 넣은 값 $want, 읽은 값 ${got:-<없음>}" >&2
  return 1
}

# 세 가지를 갈라서 본다. 예전에는 셋 다 "노드가 안 보인다" 하나로 나가서,
# 스택이 멀쩡히 떠 있는데도 조작자가 스택을 의심하게 됐다 (2026-09-10).
# ROS 가 안 걸려 있으면 **직접 건다.** start_teleop.sh · start_session.sh 와 같은
# 규약이다 — 조작자가 터미널마다 source 를 기억해야 하는 것 자체가 함정이다.
# ROS 의 setup 스크립트는 nounset 을 못 견디므로(정의 없는 변수를 참조한다)
# 이 구간에서만 -u 를 푼다.
if ! command -v ros2 >/dev/null 2>&1; then
  set +u
  [[ -f /opt/ros/jazzy/setup.bash ]] && source /opt/ros/jazzy/setup.bash
  [[ -f "$HERE/../install/setup.bash" ]] && source "$HERE/../install/setup.bash"
  set -u
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "✗ ros2 를 찾지 못했다. ROS 를 걸고 다시 실행하라:" >&2
  echo "    source /opt/ros/jazzy/setup.bash && source ~/FR5-for-RUS/install/setup.bash" >&2
  exit 1
fi
# --no-daemon 으로 본다. ros2 데몬은 오래된 목록을 들고 있을 수 있고, 그러면
# 노드가 살아 있는데도 빈 목록이 돌아온다. 대신 매번 새로 discovery 를 도므로
# 한 번쯤 놓칠 수 있어 **몇 번 다시 본다** — 한 번 놓친 것으로 스택을 의심하게
# 만들지 않기 위해서다.
seen=0
for _try in 1 2 3; do
  if ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -q "${NODE#/}"; then
    seen=1; break
  fi
done
if [[ $seen -eq 0 ]]; then
  echo "✗ $NODE 가 안 보인다." >&2
  echo "  · 스택이 떠 있는가:  pgrep -af us_diff_ik" >&2
  echo "  · 떠 있는데도 안 보이면 데몬이 오래된 것이다:  ros2 daemon stop" >&2
  echo "  · 아예 안 떠 있으면:  ./scripts/start_teleop.sh" >&2
  exit 1
fi

echo "전제조건 (진입 = 목표 − $ENTER_OFFSET_N · 이탈 $RELEASE_N · 밴드 $BAND_N)"
setp teleop.contact_probing_release_n "$RELEASE_N" || { echo "  ✗ 이탈 문턱 설정 실패" >&2; exit 1; }
setp contact_control.deadband_n "$BAND_N"          || { echo "  ✗ 밴드 설정 실패" >&2; exit 1; }
# 앞 회차가 대조군 도중 멈췄으면 힘 유지가 꺼진 채 남아 있다. 그러면 이번 회차
# 전체가 조용히 대조군이 된다 — 켜져 있는 것을 확인하고 시작한다.
setp contact_control.force_hold_enabled true       || { echo "  ✗ 힘 유지 설정 실패" >&2; exit 1; }

for t in "${TARGETS[@]}"; do
  awk -v t="$t" -v r="$RELEASE_N" -v b="$BAND_N" -v d="$ENTER_OFFSET_N" 'BEGIN{
    e = t - d
    if (r+0 >= e+0) { printf "  ✗ 목표 %s: 이탈 %s 이 진입 %.2f 이상이다\n", t, r, e; exit 1 }
    if (t-b <= r+0) { printf "  ✗ 목표 %s: 밴드 아래끝이 이탈 문턱 이하다\n", t; exit 1 }
  }' || exit 1
done
echo "  ✓ 목표 ${TARGETS[*]}  (구간: $DIRECTION)"
echo
echo "GUI 에서 준비할 것:  z (힘 영점, 공중에서) → 접촉 → c (깊이 0) → p 로 구간을 '$DIRECTION' 에 맞출 것"
echo

n=0
for t in "${TARGETS[@]}"; do
  n=$((n + 1))
  entry=$(awk -v t="$t" -v d="$ENTER_OFFSET_N" 'BEGIN{ printf "%.2f", t - d }')
  echo "▶ [$n/${#TARGETS[@]}] 목표 $t N  (진입 $entry N)"
  # 진입 문턱을 **먼저** 내린다. 목표만 올리고 문턱이 옛 값에 남아 있으면,
  # 접촉이 끊겼다 다시 붙을 때 엉뚱한 힘에서 잡힌다.
  setp teleop.contact_probing_force_n "$entry" || { echo "  ✗ 진입 문턱 설정 실패" >&2; exit 1; }
  setp contact_control.target_force_n "$t"     || { echo "  ✗ 목표 설정 실패" >&2; exit 1; }

  if [[ $n -eq 1 ]]; then
    python3 "$WAIT_SETTLED" --target "$t" --timeout-s "$SETTLE_S" --namespace "$NS" --await-contact
  else
    python3 "$WAIT_SETTLED" --target "$t" --timeout-s "$SETTLE_S" --namespace "$NS"
  fi
  case $? in
    0) ;;
    2) echo "  ✗ 접촉이 끊겼다. 다시 접촉시킨 뒤 남은 목표로 이어서 돌려라." >&2; exit 3 ;;
    *) echo "  ✗ ${SETTLE_S}s 안에 힘이 앉지 않았다." >&2
       echo "    루프가 목표 주위를 돌면 감쇠를 올려라:" >&2
       echo "      ros2 param set $NODE contact_control.admittance_b_z 3000.0" >&2
       exit 4 ;;
  esac

  echo "  ✔ 앉았다 —  GUI 창을 눌러 포커스를 주고 space 로 점을 찍어라."
  read -r -p "     찍었으면 Enter (건너뛰려면 s + Enter, 중단은 Ctrl-C): " ans
  [[ "$ans" == "s" ]] && echo "     건너뜀"
  echo
done

echo "끝. GUI 에서 s 로 저장한 뒤:"
echo "  python3 phantom_stiffness/fit_stiffness.py phantom_stiffness/runs/stiff_*  --force-band 0.5 4.0"
