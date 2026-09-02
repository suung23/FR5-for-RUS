# Sonologger manuscript — evidence map and open items

Companion to `Sonologger_Manuscript.docx` / `.pdf`. Every technical claim in the
manuscript is traced to a repository file below. Claims that could not be traced
were either removed or carried as `[TO BE CONFIRMED]`.

The manuscript text lives in exactly one place, `docs/build_sonologger_manuscript.py`,
following the same convention as the figure generators in `imu_bench/qc_track/`.
Rebuild with:

```bash
python3 docs/build_sonologger_manuscript.py
soffice --headless --convert-to pdf Sonologger_Manuscript.docx
```

Rendered length: **5 pages**, A4, 17 mm margins, 9.2 pt Times New Roman.

## 1. Evidence table

| # | Manuscript claim | Source in this repository |
|---|---|---|
| 1 | Two acquisition paths joined on one host clock; join key `pc_unix` from `time.time()` at reception | `imu_bench/host/us_imu_collect.py` (module docstring, `collect()`, `session.meta.json` writer); `imu_bench/host/us_imu_gui.py` |
| 2 | Ultrasound path GE 4C-RS → GE Voluson P6 monitor → HDMI capture (HDMI in → USB video) → host, ~8 fps | `imu_bench/qc_track/plot_fig_sonologger.py::draw_architecture` (lines 140–160); rendered in `fig_sonologger_acquisition.png` panel (b) |
| 3 | Instrumented probe build: dimensional survey → Fusion 360 CAD (`ultrasound_probe v5`) → printed hinged split clamp shell → BNO085 breakout (GY-BN008X) on bracket | `imu_bench/qc_track/plot_fig_sonologger.py` lines 245–256 and photographs in `report/figures/assets/` |
| 4 | BNO085 report rates: accelerometer 200 Hz, gyroscope 200 Hz, magnetometer 100 Hz, rotation vector 100 Hz | `imu_bench/firmware/umi_device_hardware/umi_device_hardware.ino` (`ACCEL_INTERVAL_US=5000`, `GYRO_INTERVAL_US=5000`, `MAG_INTERVAL_US=10000`, `RV_INTERVAL_US=10000`) |
| 5 | CRC-protected binary record stream over USB serial, `/dev/ttyACM0`, 115 200 baud; device time (µs) and host time both retained | `imu_bench/host/umi_protocol.py` (`crc8`, record types, `_S_*` structs); `imu_bench/host/imu_log.py` (`COLUMNS[0:2]` = `pc_unix`, `dev_us`) |
| 6 | Host fusion is Madgwick AHRS; MARG update when a magnetometer vector is supplied, IMU-only (6-axis) update otherwise; NWU earth frame; β = 0.05 | `imu_bench/host/fusion.py` (`class Madgwick`, docstring, `update()`) |
| 7 | Reference-pose calibration: `q_rel(t) = q0⁻¹ ⊗ q(t)`; earth-frame accelerometer bias `b_E`; stillness gate before accepting the reference | `imu_bench/host/zero_ref.py` (`ZeroReference.relative_quat`, `linear_accel`, `STILL_GYRO_SD=0.005`, `STILL_GYRO_MEAN=0.005`, `STILL_ACCEL_SD=0.15`) |
| 8 | Per-sample log columns (36) incl. raw/calibrated magnetometer, host quaternion, chip quaternion, `rel_roll/pitch/yaw`, BNO085 calibration status | `imu_bench/host/imu_log.py::COLUMNS` |
| 9 | Session store layout: `imu_<stamp>.csv`, `imu_<stamp>.meta.json`, `us_frames.bin`, `us_index.csv` (`pc_unix, us_seq, frame_id, byte_offset`), `session.meta.json` | `imu_bench/host/us_imu_collect.py` docstring and `finally:` block; verified against `imu_bench/logs/us_imu_*/` |
| 10 | Longest recorded synchronized session: 776 ultrasound frames over 96.9 s = 8.00 fps, median inter-frame interval 125.9 ms; 24 471 IMU rows over 97.0 s = 252.4 Hz | measured directly from `imu_bench/logs/us_imu_20260819_202202/us_index.csv` and `imu_20260819_202202.csv` |
| 11 | Five recorded sessions; three carry both streams; frames are 256 × 256 uint8 stored without transformation | `imu_bench/logs/us_imu_*/session.meta.json` (`frame_shape [256,256]`, `dtype uint8`, note "candidate — 방향/scan conversion 미검증") |
| 12 | Ultrasound fixed end-to-end latency is **not** removed by arrival-time alignment and has not been measured | `us_imu_collect.py` docstring; `session.meta.json` `caveat` field; `DESIGN_NOTES.md` §11.1 (US frame-grabber latency marked ⏳) |
| 13 | Orientation QC protocol: instrumented probe on FR5 flange, robot forward kinematics as ground truth, teleoperated probing-like rhythm (move → hold ≥ 1.5 s → move), 90 s `probe` block, separate `probe_cal` block for validation | `imu_bench/qc_track/README.md`; `imu_bench/qc_track/protocol.py` (`PROMPTS["probe"]`, `BLOCKS`, `DURATION`) |
| 14 | Time-base health for the `probe` block: paired fraction 1.00, clock skew −25 ppm, pairing residual p95 4.93 ms, GT 144 Hz / IMU 253 Hz | `imu_bench/qc_track/raw_data/meta/qc_result.json` → `timebase.probe` |
| 15 | Effective latency −22.8 / −21.1 / −6.6 ms (tilt x, tilt y, axial rotation); cross-correlation 0.997 / 0.999 / 1.000 | `qc_result.json` → `rotation.host6.channels.*.latency_ms`, `.corr` |
| 16 | Latency-corrected angular RMS error 1.14° / 0.73° / 1.95° per channel, 0.42° tilt magnitude, 2.50° geodesic (p95 3.56°) | `qc_result.json` → `rotation.host6.channels.*.err_rms_delay_corrected_deg`, `rotation.host6.geodesic.rms_delay_corrected_deg` |
| 17 | 6-axis vs on-chip 9-axis rotation vector: 1.95 vs 11.23° (×5.8), 1.14 vs 6.17° (×5.4), 0.73 vs 3.98° (×5.4), 0.42 vs 0.70° (×1.7) | `qc_result.json` → `rotation.host6` / `rotation.chip`; figure built by `plot_fig_6ax_vs_9ax.py::build` |
| 18 | Static alignment residual RMS 0.81° (6-axis, 7 stationary poses, observability 1.05) vs 6.68° (9-axis) | `qc_result.json` → `alignment.resid_rms_deg`; `qc_chip.json` → `alignment.resid_rms_deg` |
| 19 | Heading (axial-rotation) drift 0.23 °/min (6-axis) vs 8.29 °/min (9-axis) | `qc_result.json` → `rotation.{host6,chip}.channels.spin.drift_deg_per_min` |
| 20 | Magnetic disturbance near the robot: pose-paired test found 0 usable hold pairs in three blocks; residual/noise up to 9.6 and direction error up to 56.8° for the fitted field model | `qc_result.json` → `mag_position.*.n_pairs`, `mag_model.probe_cal` |
| 21 | Still-segment displacement error floor with zero-bias + ZUPT: 0.25 / 0.67 / 1.97 / 6.32 mm at 0.5 / 1 / 2 / 5 s windows; raw double integration 12.4 / 49.5 / 197.8 / 1236.5 mm | `qc_result.json` → `translation.mag_map.error_floor_mm` (stages A and C) |
| 22 | Per-segment displacement error over 21 move segments is one to two orders of magnitude above the still-segment floor | `qc_result.json` → `translation.probe.segments[*].eps_mm`, `n_segments = 21` |
| 23 | Independent bench reference for the displacement floor (0.23 / 0.67 / 1.7 / 7 mm) | `imu_bench/QC_PLAN.md` item 9; `imu_bench/qc_track/reference.py` (`displacement_floor_mm`) |
| 24 | Pre-registered pass criteria and the recorded verdict: dynamic attitude RMS 2.50° exceeds the 2.0° limit; alignment, latency, correlation, drift and displacement-floor criteria pass | `qc_result.json` → `verdict`; `imu_bench/qc_track/reference.py::PASS` |
| 25 | Accelerometer scale-error dependence on chip calibration state (+2.28 % uncalibrated vs +0.16 % calibrated) | `imu_bench/qc_track/reference.py` → `scale_error_pct`; `QC_PLAN.md` item 8 |
| 26 | Clinical direction: non-interventional observational recording during HoLEP morcellation bladder ultrasound monitoring; existing workflow preserved; robot, F/T sensor and automatic contact-force control **not** applied to research subjects; IRB application in preparation | `pdfs/tips_ultrasound_robot_clinical_study_v03.pdf` (page text, incl. the sentence "임상 데이터 수집 단계에서는 … 로봇·F/T 센서·자동 접촉력 제어는 연구대상자에게 적용하지 않음") |
| 27 | `T_probe(t) ∈ SE(3)` is named as a *later* control-stage representation, not something the IMU supplies | same PDF ("임상 수집 · IMU 기반 상대 회전 정보 기록" / "후속 제어 단계 · pose state representation 으로 확장") |
| 28 | Downstream learning workflow (ZUPT-bracketed action chunks, ACT CVAE with quality and force heads, online control stack) is a design, not a trained result | `imu_bench/qc_track/plot_fig_learning_control.py` docstring ("DESIGN_NOTES.md §3 … ASCII 도식을 그대로 옮긴 것"); `DESIGN_NOTES.md` §3 |
| 29 | Probe geometry constants used in the text (GE 4C-RS convex array) | `teleop_gui/src/components/ProbeAssembly.tsx`; `fr5_control/fr5_control/config/probe.yaml`; `docs/plot_fig_probe_frame.py` |

## 2. Figures used

| Manuscript | File | Size (px) | Generator |
|---|---|---|---|
| Figure 1 | `imu_bench/qc_track/report/figures/fig_sonologger_acquisition.png` | 4980 × 3120 | `imu_bench/qc_track/plot_fig_sonologger.py` |
| Figure 2 | `imu_bench/qc_track/report/figures/fig_6axis_rotation_translation.png` | 4500 × 3180 | `imu_bench/qc_track/plot_fig_6ax_rot_trans.py` |
| Figure 3 | `imu_bench/qc_track/report/figures/fig_learning_to_control.png` | 4920 × 3060 | `imu_bench/qc_track/plot_fig_learning_control.py` |

Existing assets were used unchanged; no figure was regenerated, redrawn or cropped.
Figure 1 covers the required "system overview" and "probe/acquisition hardware"
categories, Figure 2 the "representative ultrasound–orientation sequence and data
characterisation" category, and Figure 3 the "downstream learning-data concept"
category, which it is captioned as a planned workflow rather than a result.

Available but **not** used, with reason:

* `fig_imu6_vs_imu9_accuracy.png` — six-axis versus nine-axis attitude comparison
  (`plot_fig_6ax_vs_9ax.py`). Included in an earlier draft as Figure 3 and then
  removed: every number it plots is already stated in Section 3.3, and it is not one
  of the four figure categories the paper needs. Removing it was what brought the
  rendered manuscript from six pages to five. It is the obvious figure to restore if
  a page budget allows.
* `fig_twist_probe_tissue.png` — probe-frame twist taxonomy. Belongs to the control
  paper, not to an acquisition-platform paper; omitted for the five-page limit.
* `figure2_pfus1_segmentation_examples.pdf`, `figure1_acquisition_domain_shift.pdf`
  (`Unet_seg/`) — segmentation performance on the PFUS1 dataset. Including them
  would imply a perception result that this platform paper does not claim.
* `docs/figures/fig_probe_frame_axes.png` — probe frame and roll/pitch/yaw
  definition. Useful but superseded by the compact frame description in the text.

## 3. Open items — `[TO BE CONFIRMED]` in the manuscript

1. **Authors, affiliations, corresponding author, ORCIDs.** No author metadata
   exists anywhere in the repository. Git author is `Suung23 <rosotarun@gmail.com>`,
   which is not a usable byline. All name fields are placeholders.
2. **HDMI capture device model and captured display resolution.** The architecture
   figure and its generator specify the path ("HDMI input → USB video, captured
   display frames, ~8 fps") but no product model, driver, or capture resolution
   appears in any code, configuration file, or log.
3. **Sessions recorded through the HDMI capture path.** Every synchronized session
   currently stored in `imu_bench/logs/` was recorded with the host collector
   reading 256 × 256 frames from a networked research image front end
   (`us_imu_collect.py --host 192.168.1.1`). The host-side timestamping and join
   logic are the same for either front end, but no logged session in this
   repository was captured through the Voluson HDMI path. The manuscript states
   this explicitly rather than glossing it.
4. **Ultrasound end-to-end (display → capture → host) latency.** Declared
   unmeasured in `session.meta.json` and in `DESIGN_NOTES.md`. Reported as an open
   quantity, not estimated.
5. **Number of operators, number of clinical sessions, participant counts.** None
   exist. The demonstration section reports bench sessions only.
6. **IRB / ethics approval number and institution.** The internal document states
   the application is in preparation with a target approval date; no approval
   number exists.
7. **Funding source and grant number.**
8. **Data and code availability statement** (public repository URL or access policy).
9. **Full bibliographic details for two clinically motivating references**
   (references 5 and 6: Robles et al., *Journal of Endourology* 2025; Tzou et al.,
   *Urology* 2020). Only surname, venue and year are recorded in
   `pdfs/tips_ultrasound_robot_clinical_study_v03.pdf`. Author initials, title,
   volume, pages and DOI could not be verified from the workspace, so the entries
   carry a placeholder rather than invented detail — note in particular that **no
   author initials are supplied for these two references**, because guessing them
   would be fabricating an author name.

## 4. Claims deliberately excluded (no supporting evidence)

* Any trained robotic ultrasound policy, imitation-learning result, or
  policy-evaluation metric. `DESIGN_NOTES.md` §16 records that `Unet_seg/data/` is
  empty and that no trained checkpoint exists; the learning stack in Figure 4 is a
  design.
* Any statement that the IMU provides translation, absolute position, or full
  SE(3) probe pose. The measured displacement error floor and the per-segment
  errors in `qc_result.json` contradict it, and the manuscript uses them to say so.
* Any clinical accuracy, efficacy, patient-contact, or autonomous-control claim.
* Any measured synchronization error between the ultrasound and inertial streams.
  What exists is (i) the arrival-time join design, (ii) IMU-versus-robot pairing
  residuals from the QC run, and (iii) an explicit statement that the fixed
  ultrasound path delay is unmeasured. These are reported separately and never
  merged into a single "synchronization accuracy" number.
* Any claim about human freehand scanning fidelity. All orientation accuracy
  numbers come from robot-actuated motion with forward-kinematics ground truth;
  the manuscript says so in the text and in the figure captions.
* Segmentation, image-quality, or force-control performance.
