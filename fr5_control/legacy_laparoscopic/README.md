# legacy_laparoscopic

복강경 봉합 리그 전용 코드. **초음파(RUS) 제어에는 사용하지 않는다.**

`COLCON_IGNORE`가 있으므로 `colcon build`가 이 디렉터리를 건너뛴다.
삭제하지 않고 남겨둔 이유는 기하 상수, 캘리브레이션 절차, ACT 추론 노드 구조를
참조할 일이 남아 있기 때문이다. 참조가 끝나면 삭제해도 된다.

전환 근거는 리포 루트 `DESIGN_NOTES.md` 참조.

## 분리된 것

### 패키지 전체

| 패키지 | 사유 |
|---|---|
| `fr5_inference/` | 봉합 태스크 ACT policy 노드 5종. 태스크가 다르다. 단 chunk delta 발행 구조는 초음파 policy 노드 템플릿으로 참조 가치가 있다 |
| `laparo_umi/` | 시리얼 UMI 장치. 복강경 기구 전용 |
| `suturing_interfaces/` | 봉합용 srv 정의. `us_interfaces`로 대체 예정 |

### `fr5_ik/` — RCM 계열 전부

```
rcm_control_node.py
rcm_two_twist.py
rcm_two_pos.py
rcm_two_delta.py
rcm_two_delta_new_rcm.py
rcm_two_absolute_new_rcm.py
calc_gripper_wrt_new_rcm_node.py
free_control_node.py          # 이미 deprecated 상태였음
freespace_two_pos.py          # pos 인터페이스 미사용
```

RCM(트로카 고정점 구속)은 초음파에 대응물이 없다. 프로브는 고정점을 통과하지 않고
피부 표면에 접촉한다. 가상 구속이 필요하다면 그것은 표면 법선 정렬이고,
Mx/My 기반 자세 정렬로 처리한다 (DESIGN_NOTES §5).

`freespace_two_twist.py`만 활성 트리에 남아 `diff_ik_node`의 베이스가 된다.

### `fr5_control/`

```
fr5_dual_direct_control_node.py
fr5_dual_direct_joint_control_node.py
fairino_driver.py             # AuxServoSetParam 한 줄짜리 스텁
```

듀얼암 전용 변종. 제어 대상은 오른팔 단일(`192.168.58.3`)이다.

### `fr5_vision/` — eye-to-hand 내시경 캘리브레이션 일체

```
calibrate_intrinsics.py
save_calibration_data.py
calculate_calibration.py
display_3d_to_2d_node.py
gt_sparse_depth_node.py
3d_to_2d.py
README_calibration.md
calibration_data/             # 내시경 intrinsics
new_camera_calibration_data/
calibration_data_right/       # charuco 31세트 + pose 31세트 + eye_to_hand_result
```

내시경은 외부 관찰자이므로 eye-to-hand 캘리브레이션을 쓴다. 초음파 프로브는
영상면이 프로브에 강체 고정된 도구 그 자체이며, probe-to-image 캘리브레이션
(N-wire / cross-wire 팬텀)으로 절차 자체가 다르다. 재사용 불가.

### `fr5_launch/`

```
inference_absolute_new_rcm.launch.py
inference_delta.launch.py
inference_delta_new_rcm.launch.py
inference_new_rcm_chunk_delta.launch.py
```

`teleop.launch.py`는 활성 트리에 남아 개조 대상이다.

## 참조 가치가 있는 것

| 대상 | 참조 이유 |
|---|---|
| `fr5_ik/*.py`의 툴 변환 행렬 | J6→툴 기하의 기존 값. 프로브 마운트 CAD 도착 시 대조용. 단 파일마다 0.64 / 0.615 / 0.22로 불일치하므로 그대로 믿으면 안 된다 |
| `fr5_inference/suturing_master_new_rcm_chunk_delta.py` | chunk delta 액션 발행 구조 |
| `gt_sparse_depth_node.py` | 로봇 기구학 → 영상 좌표 투영 패턴 |

## 아직 분리하지 않은 것

판단이 필요해 활성 트리에 남겨둔 항목:

- `dataset/annotate/` — 복강경 sparse depth 어노테이션 도구(`fix_sparse_depth.py` 등)와
  일반 h5 도구가 섞여 있다
- `dataset/test/`, `dataset/dataset_visualizer/` — 복강경 에피소드 h5와 추출 이미지 178장
- `keyboard_teleop` — `/desired_pose_wrt_rcm`을 발행하므로 twist 기반으로 개조 필요
