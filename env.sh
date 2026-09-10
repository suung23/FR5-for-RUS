# 정책 실행 환경 — ROS 를 먼저, venv 를 나중에.
#
#   source ~/FR5-for-RUS/env.sh
#
# torch 는 venv 에, rclpy 는 ROS 에 있다. setup.bash 가 PYTHONPATH 와 LD_LIBRARY_PATH 를
# 둘 다 잡아 주므로 ROS 를 먼저 소스해야 하고 (PYTHONPATH 만으로는 librcl_action.so 를
# 못 찾는다), venv 는 그 뒤에 켜야 python3 가 venv 것으로 잡힌다. 시스템 파이썬과 venv 가
# 둘 다 3.12 라서 한 프로세스에서 공존한다.
# 이미 켜진 venv 가 있으면 **먼저 끈다.** activate 는 _OLD_VIRTUAL_PATH 를 되살리는데
# 그 값은 "처음 켤 때의 PATH" 라, 그 뒤에 소스한 ROS 가 통째로 지워진다. 그러면 이 파일이
# "환경 OK" 를 찍고도 ros2 가 없는 상태가 된다 (2026-09-11 실측).
if type deactivate >/dev/null 2>&1; then deactivate; fi

source /opt/ros/jazzy/setup.bash
source ~/FR5-for-RUS/install/setup.bash 2>/dev/null || \
    echo "⚠️  install/setup.bash 가 없다 — colcon build 를 먼저 하십시오"
source ~/FR5-for-RUS/Unet_seg/.venv/bin/activate
python3 - <<'PY' || echo "⚠️  환경이 아직 맞지 않습니다"
import rclpy, torch
print(f"환경 OK — rclpy · torch {torch.__version__} (CUDA {torch.cuda.is_available()})")
PY
