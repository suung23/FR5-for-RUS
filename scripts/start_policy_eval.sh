#!/usr/bin/env bash
#
# 정책 평가 — 힘 탐색과 러너를 함께 띄운다.
#
#   ./scripts/start_policy_eval.sh                          # 기본: 루프 평가, 회전만, 3 °/s
#   ./scripts/start_policy_eval.sh --dry                    # 지령 없이 (축·중력·지각 확인)
#   ./scripts/start_policy_eval.sh --pilot                  # 파일럿 6 자세 × 3 조건 = 18
#   ./scripts/start_policy_eval.sh --main 25                # 본 시험 25 자세 × 4 조건
#   ./scripts/start_policy_eval.sh --pilot --conditions hold,search   # 조건을 바꿔서
#   ./scripts/start_policy_eval.sh -- --condition search    # 실험 아님 — 탐색만 한 번 돌려 본다
#   ./scripts/start_policy_eval.sh --out runs/eval_2        # 기록 위치
#   ./scripts/start_policy_eval.sh -- --max-deg-s 5         # `--` 뒤는 러너로 그대로
#
# 루프 평가와 실험을 **같은 명령이 띄우는** 이유. 실험(run_experiment.py)도 힘 탐색이
# 없으면 설정값이 출발값에 고정된다 — 루프 평가와 똑같이. 예전에는 이 스크립트가
# run_policy.py --loop 만 띄워서, 파일럿을 돌리려면 다시 터미널을 셋으로 벌려야 했고
# 거기서 힘 탐색이 빠지면 그 에피소드들은 나중에 분리해 보고해야 한다.
#
# 파일럿·본시험의 --out 은 **시각이 안 붙는다.** 끊긴 세션을 같은 폴더로 이어서 하기
# 때문이다 (PROTOCOL_EXPERIMENT.md §4).
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
OUT=""
EXECUTE="--execute"
MODE="loop"          # loop | pilot | main
POSES=""
CONDITIONS=""        # 비면 파일럿 hold,placebo,policy · 본시험 +expert
EXTRA=()

while (( $# )); do
  case "$1" in
    --dry)        EXECUTE=""; shift ;;
    --pilot)      MODE="pilot"; shift ;;
    --conditions) CONDITIONS="${2:-}"; shift; (( $# )) && shift ;;
    # `shift 2` 는 인자가 하나뿐일 때 **실패하고 아무것도 안 옮긴다.** set -e 가 없으므로
    # 루프는 계속 돌고 같은 case 가 다시 잡힌다 — 무한 루프다. 값이 있든 없든 한 번은
    # 반드시 옮기고, 값이 있을 때만 한 번 더 옮긴다.
    --main)       MODE="main"; POSES="${2:-}"; shift; (( $# )) && shift ;;
    --ckpt)       CKPT="${2:-}"; shift; (( $# )) && shift ;;
    --out)        OUT="${2:-}"; shift; (( $# )) && shift ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    sed -n '2,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

# 본 시험의 자세 수는 파일럿 분석이 정한다. 숫자가 아니면 여기서 멈춘다 —
# 예전에 `--poses <분석이 알려준 N>` 을 그대로 붙여넣어 bash 가 리다이렉션으로 읽은
# 적이 있다. 자리표시자는 명령이 아니다.
if [[ "$MODE" == "main" ]]; then
  if ! [[ "$POSES" =~ ^[0-9]+$ ]]; then
    echo "--main 뒤에 자세 수를 숫자로 주십시오 (예: --main 25)." >&2
    echo "  몇 개인지는 파일럿 분석이 알려준다:" >&2
    echo "    python3 policy_learning/scripts/analyze_experiment.py policy_learning/runs/pilot" >&2
    exit 1
  fi
fi

if [[ -z "$OUT" ]]; then
  case "$MODE" in
    pilot) OUT="$WORKSPACE/policy_learning/runs/pilot" ;;
    main)  OUT="$WORKSPACE/policy_learning/runs/main" ;;
    *)     OUT="$WORKSPACE/policy_learning/runs/eval_$(date +%Y%m%d_%H%M%S)" ;;
  esac
fi

# ROS 의 setup 스크립트는 **nounset 을 견디지 못한다** — /opt/ros/jazzy/setup.bash 8 행이
# AMENT_TRACE_SETUP_FILES 를 확인 없이 읽는다. set -u 아래에서 source 하면 그 자리에서
# 셸이 죽고, 이 스크립트는 2>&1 로 그것을 버리고 있었으므로 **아무것도 찍지 않고 rc=1 로
# 끝났다.** 조용한 실패라 "왜 안 뜨지" 가 로그 어디에도 남지 않는다.
# start_session.sh 는 같은 자리를 set +u 로 감싸고 있다 — 여기도 맞춘다.
#
# 진단 메시지도 더 이상 버리지 않는다. 실패하면 그 출력이 유일한 단서다.
set +u
# shellcheck source=../env.sh
if ! source "$WORKSPACE/env.sh" >/tmp/.rus_env.$$ 2>&1; then
  echo "env.sh 를 못 읽었다:" >&2
  cat /tmp/.rus_env.$$ >&2
  rm -f /tmp/.rus_env.$$
  exit 1
fi
rm -f /tmp/.rus_env.$$
set -u

[[ -f "$CKPT" ]] || { echo "체크포인트가 없다: $CKPT"; exit 1; }

# 제어 스택이 떠 있어야 지령이 갈 데가 있다. 없는 채로 띄우면 러너는 조용히 대기만 한다.
#
# **한 번만 보고 포기하지 않는다.** ROS 2 디스커버리는 비동기라, 세션이 막 뜬 직후에는
# 노드가 실제로 살아 있어도 `ros2 node list` 에 아직 안 나온다. 2026-09-11 에 세션을
# 띄우자마자 이 명령을 친 조작자가 "안 보인다" 를 받았는데 노드는 멀쩡히 돌고 있었다.
# 없다고 단정하려면 기다려 봐야 한다.
WAIT_S=15
echo -n "제어 스택 확인"
for ((i = 0; i < WAIT_S; i++)); do
  if ros2 node list 2>/dev/null | grep -q us_diff_ik_node; then
    echo "  us_diff_ik_node ✓"
    break
  fi
  echo -n "."
  sleep 1
done
if ! ros2 node list 2>/dev/null | grep -q us_diff_ik_node; then
  echo
  echo "⚠️  us_diff_ik_node 가 ${WAIT_S} s 동안 안 보인다 — 세션을 먼저 띄우십시오:"
  echo "      ~/FR5-for-RUS/scripts/start_session.sh"
  echo "    떠 있는데도 이 메시지가 나오면 남은 노드를 정리하고 다시 띄우십시오:"
  echo "      ~/FR5-for-RUS/scripts/stop_all.sh"
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
case "$MODE" in
  pilot) echo "실험        파일럿 — 6 자세 × ${CONDITIONS:-hold,placebo,policy} , 90 s, 눈가림" ;;
  main)  echo "실험        본 시험 — $POSES 자세 × 4 조건 = $((POSES * 4)) 에피소드, 90 s, 눈가림" ;;
  *)     echo "실험        아님 (자유 루프 평가)" ;;
esac
if [[ -d "$OUT" ]] && [[ "$MODE" != "loop" ]]; then
  echo "            ↑ 이미 있다 — 끊긴 세션을 **이어서** 한다"
fi
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
case "$MODE" in
  pilot)
    python3 scripts/run_experiment.py "$CKPT" --out "$OUT" \
            --poses 6 --conditions "${CONDITIONS:-hold,placebo,policy}" \
            --duration 90 --blind $EXECUTE \
            --axes rot --max-deg-s 3 ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  main)
    python3 scripts/run_experiment.py "$CKPT" --out "$OUT" \
            --poses "$POSES" --conditions "${CONDITIONS:-hold,placebo,policy,expert}" \
            --duration 90 --blind $EXECUTE \
            --axes rot --max-deg-s 3 ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
  *)
    python3 scripts/run_policy.py "$CKPT" --loop $EXECUTE \
            --axes rot --max-deg-s 3 --out "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
esac
