#!/usr/bin/env bash
#
# 지금 노드가 들고 있는 값과 불변식을 찍는다. 실험 전에 이걸 먼저 본다.
#
#   ./check_params.sh                    # 기본 노드
#   ./check_params.sh /us_diff_ik_node   # 노드 지정
#
# 값을 못 읽으면 조용히 넘어가지 않는다 — 빈 값으로 계산을 이어 가면 "불가"인지
# "모른다"인지 구별할 수 없고, 그 구별이 여기서 가장 중요하다.
set -uo pipefail

NODE=${1:-/us_diff_ik_node}
TARGETS=${TARGETS:-"0.5 1.0 1.5 2.0 3.0 4.0"}

command -v ros2 >/dev/null || {
  echo "✗ ros2 가 없다.  source /opt/ros/jazzy/setup.bash && source ~/FR5-for-RUS/install/setup.bash" >&2
  exit 1; }

if ! ros2 node list 2>/dev/null | grep -qx "$NODE"; then
  echo "✗ $NODE 가 안 보인다. 지금 떠 있는 노드:" >&2
  ros2 node list 2>/dev/null | sed 's/^/    /' >&2 || true
  echo "  → 터미널 1 에서 ./scripts/start_session.sh 를 띄웠는가?" >&2
  exit 1
fi

PARAMS=(
  teleop.contact_probing_force_n
  teleop.contact_probing_release_n
  contact_control.deadband_n
  contact_control.target_force_n
  safety.warn_contact_force_n
  safety.max_contact_force_n
)

declare -A VAL
fail=0
for p in "${PARAMS[@]}"; do
  raw=$(ros2 param get "$NODE" "$p" 2>&1)
  v=$(printf '%s' "$raw" | awk '/value is:/ {print $NF}')
  if [[ -z "$v" ]]; then
    printf '  %-34s ✗ 못 읽었다 — %s\n' "$p" "$(printf '%s' "$raw" | head -1)"
    fail=1
  else
    printf '  %-34s %s\n' "$p" "$v"
  fi
  VAL["$p"]="$v"
done

if (( fail )); then
  echo
  echo "✗ 파라미터를 못 읽었다. 이름이 바뀌었거나 노드가 다른 것이다." >&2
  echo "  전체 목록:  ros2 param list $NODE" >&2
  exit 1
fi

echo
python3 - "${VAL[teleop.contact_probing_force_n]}" \
         "${VAL[teleop.contact_probing_release_n]}" \
         "${VAL[contact_control.deadband_n]}" \
         "${VAL[safety.warn_contact_force_n]}" \
         "${VAL[safety.max_contact_force_n]}" "$TARGETS" <<'PY'
import sys
enter, rel, band, warn, mx = (float(v) for v in sys.argv[1:6])
targets = [float(t) for t in sys.argv[6].split()]

print(f"{'목표':>6}  {'진입<목표':<10}{'목표-밴드>이탈':<16}{'목표<경고':<10}결론")
bad = []
for t in targets:
    a, b, c = enter < t, t - band > rel, t < warn
    if not (a and b and c):
        bad.append(t)
    print(f"{t:6.1f}  {str(a):<10}{str(b):<16}{str(c):<10}"
          f"{'가능' if a and b and c else '❌ 불가'}")

if bad:
    print(f"\n❌ 도달할 수 없는 대역: {', '.join(f'{t:g}' for t in bad)} N")
    if any(enter >= t for t in bad):
        print(f"   진입 문턱 {enter:g} N 이 목표 이상이다 — 접촉해도 힘 제어가 켜지지 않는다.")
    print("\n   걸어야 할 값:")
    print("     ros2 param set /us_diff_ik_node teleop.contact_probing_release_n 0.10")
    print("     ros2 param set /us_diff_ik_node teleop.contact_probing_force_n 0.20")
    print("     ros2 param set /us_diff_ik_node contact_control.deadband_n 0.05")
    print("\n   (run_repeat.sh 는 이제 이 셋을 스스로 건다.)")
    sys.exit(1)
print(f"\n✅ 대역 {len(targets)} 개 모두 도달 가능하다.")
PY
