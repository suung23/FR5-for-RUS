#!/usr/bin/env bash
#
# 정책 평가 — 힘 탐색과 러너를 함께 띄운다.
#
#   ./scripts/start_policy_eval.sh                          # 기본: 루프 평가, 회전만, 3 °/s
#   ./scripts/start_policy_eval.sh --dry                    # 지령 없이 (축·중력·지각 확인)
#   ./scripts/start_policy_eval.sh --out runs/eval_2        # 기록 위치
#   ./scripts/start_policy_eval.sh -- --max-deg-s 5         # `--` 뒤는 러너로 그대로
#
# 둘을 따로 띄우면 **힘 탐색을 빠뜨리기 쉽고**, 그러면 힘 설정값이 출발값에 고정된 채
# 돈다 (Q_raw 기반 제어가 죽은 상태). 러너가 그것을 경고하긴 하지만, 경고를 보고 다른
# 터미널을 찾는 것보다 처음부터 같이 뜨는 편이 낫다.
#
# Ctrl-C 하나로 둘 다 내린다. 힘 탐색만 남아 설정값을 계속 옮기는 상태를 막는다.
set -uo pipefail

cd "$(dirname "$0")/.."
WORKSPACE="$(pwd)"

CKPT="$WORKSPACE/policy_learning/runs/qres2_ep25.pt"
OUT="$WORKSPACE/policy_learning/runs/eval_$(date +%Y%m%d_%H%M%S)"
EXECUTE="--execute"
EXTRA=()

while (( $# )); do
  case "$1" in
    --dry)        EXECUTE=""; shift ;;
    --ckpt)       CKPT="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

# shellcheck source=../env.sh
source "$WORKSPACE/env.sh" >/dev/null 2>&1 || { echo "env.sh 를 못 읽었다"; exit 1; }

[[ -f "$CKPT" ]] || { echo "체크포인트가 없다: $CKPT"; exit 1; }

# 제어 스택이 떠 있어야 지령이 갈 데가 있다. 없는 채로 띄우면 러너는 조용히 대기만 한다.
if ! ros2 node list 2>/dev/null | grep -q us_diff_ik_node; then
  echo "⚠️  us_diff_ik_node 가 안 보인다 — ./scripts/start_session.sh 를 먼저 띄우십시오"
  exit 1
fi

mkdir -p "$OUT"
FS_LOG="$OUT/force_search.log"

cleanup() {
  [[ -n "${FS_PID:-}" ]] && kill "$FS_PID" 2>/dev/null
  wait "${FS_PID:-}" 2>/dev/null
  echo
  echo "힘 탐색 로그: $FS_LOG"
}
trap cleanup EXIT INT TERM

echo "체크포인트  $CKPT"
echo "기록        $OUT"
echo "모드        ${EXECUTE:-DRY-RUN (지령 없음)}"
echo

# ① 힘 탐색 — 설정값을 Q_raw 경사로 옮긴다. --dry 면 관찰만.
FS_ARGS=(--ros-args -p "execute:=$([[ -n "$EXECUTE" ]] && echo true || echo false)")
ros2 run fr5_control force_search "${FS_ARGS[@]}" >"$FS_LOG" 2>&1 &
FS_PID=$!
sleep 2
if ! kill -0 "$FS_PID" 2>/dev/null; then
  echo "힘 탐색이 즉시 죽었다 — $FS_LOG 를 보십시오"; tail -5 "$FS_LOG"; exit 1
fi
echo "① 힘 탐색 pid $FS_PID"

# ② 러너 — 앞에서 돈다. Ctrl-C 가 여기로 오고, trap 이 ① 을 내린다.
echo "② 러너 (Ctrl-C 로 둘 다 종료)"
echo
cd "$WORKSPACE/policy_learning"
python3 scripts/run_policy.py "$CKPT" --loop $EXECUTE \
        --axes rot --max-deg-s 3 --out "$OUT" "${EXTRA[@]}"
