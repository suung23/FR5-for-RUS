# 학습된 면내 정책 (참고용)

**본 시험(`MAIN_RESULT.md`)에는 이 모델이 들어 있지 않다.** 그 시험은 규칙 기반 제어기를
잰 것이다. 이 모델은 프리핸드 시연에서 면내 방향을 배울 수 있는지를 확인한 산출물이고,
후속 시험(`docs/PROTOCOL_SONOLOGGER_INPLANE.md`)의 재료로 남긴다.

| 파일 | 무엇 |
|---|---|
| `inplane_policy_final_v2.pt` | 체크포인트 (ACT CVAE, 2.69 M 파라미터, epoch 44) |
| `inplane_policy_final_v2.pt.action_bias.json` | Q̂ 의 관측 무관 행동 기울어짐 보정 |
| `inplane_policy_final_v2.config.yaml` | 학습 설정 |

## 성능 (보유하지 않은 검증 세션)

| | 값 |
|---|---|
| `select_corr_dx_thy` (면내 방향, 음수가 옳다) | **−0.809** |
| `select_dir_thy` | 0.811 |
| `select_nmae` | 15.38 |
| θy MAE | 1.68° |

비교: 부호 수정 전 정책의 면내 상관은 **+0.178** 이었다.

## 재현

데이터셋(`data/policy_dataset.h5`, 5.8 GB)은 리포에 없다. 있는 기계에서:

```bash
python3 scripts/train_policy.py --out runs/final_v2 \
    --set train.num_workers=8 --set train.checkpoint_metric=select_corr_dx_thy \
    --set train.epochs=70 --set train.checkpoint_min_epoch=10
python3 scripts/measure_action_bias.py runs/final_v2/best.pt
```

`checkpoint_min_epoch=10` 이 없으면 `best.pt` 가 epoch 1 에 찍힌다 — β 워밍업 중에는
잠재변수가 거의 안 쓰여 방향 상관이 과장되기 때문이다 (epoch 1 이 corr −0.904 로 1 등인데
`select_nmae` 15.6 · θy 2.54°).
