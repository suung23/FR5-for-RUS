#pragma once
// ============================================================================
//  UMI device — per-board calibration
// ----------------------------------------------------------------------------
//  Every physical device differs slightly: sensor mounting angle, magnet
//  strength/position on the gripper, and the rod insertion geometry. Keeping
//  those numbers HERE (out of the firmware logic in umi_device_hardware.ino)
//  means the same sketch can be flashed to any board just by pointing the
//  build at that board's profile — no edits to the .ino needed.
//
//  This file is plain text and lives in the sketch folder, so the Arduino
//  toolchain (Arduino IDE or arduino-cli) automatically includes it when it
//  compiles and uploads umi_device_hardware.ino.
//
//  HOW TO CALIBRATE A NEW BOARD
//    1. Flash umi_print_distance_value/ and read the raw mm at the fully
//       inserted and fully retracted rod positions -> CAL_DIST_RAW_MIN/MAX_MM.
//    2. Flash umi_print_hallsensor_value/ and read the raw 12-bit ADC value at
//       0 / 25 / 50 / 100 % gripper closure -> CAL_HALL_ADC_*. The ADC may rise
//       or fall as the gripper closes depending on the magnet's polarity — just
//       record whatever the sensor actually reads at each closure point. The
//       firmware handles both directions automatically (see hallToPercent()).
//    3. Copy a profile block below, bump the UMI_BOARD_ID number, and paste in
//       the measured values.
//
//  SELECTING THE ACTIVE BOARD
//    Option A (simplest): edit the default UMI_BOARD_ID below.
//    Option B (no edits to this file — good for CI / multiple boards):
//       arduino-cli compile --fqbn <fqbn> \
//         --build-property "compiler.cpp.extra_flags=-DUMI_BOARD_ID=2" \
//         umi_device_hardware
//       arduino-cli upload  --fqbn <fqbn> -p <port> umi_device_hardware
//       (NOTE: build.extra_flags 를 덮어쓰면 플랫폼의 MCU 정의가 사라져
//        "Unsupported MCU" 로 빌드가 깨진다 — compiler.cpp.extra_flags 를 쓸 것)
// ============================================================================

#ifndef UMI_BOARD_ID
#define UMI_BOARD_ID 6        // <-- default board used when none is passed on the command line
#endif

// Output mode: 0 = binary dual-stream protocol (default),
//              1 = legacy 7-field CSV only, now at 100Hz (old host tools).
// Can be overridden from the command line like UMI_BOARD_ID:
//   --build-property "build.extra_flags=-DLEGACY_CSV=1"
#ifndef LEGACY_CSV
#define LEGACY_CSV 0
#endif

// Hardware population: 1 = IMU-only board (BNO085 alone on the bus — no
// VL53L0X ToF, no hall sensor). Their bring-up, sampling and records are
// compiled out; the FUSED record still carries the rotation vector, with
// dist/grip left at their idle values. Overridable like UMI_BOARD_ID:
//   --build-property "compiler.cpp.extra_flags=-DUMI_IMU_ONLY=1"
#ifndef UMI_IMU_ONLY
#define UMI_IMU_ONLY 0
#endif


// ----------------------------------------------------------------------------
//  Common to all boards (hardware wiring — rarely changes between identical units)
// ----------------------------------------------------------------------------
#define BNO08X_INT_PIN     D9
#define BNO08X_RST_PIN     D10
#define HALL_SENSOR_PIN    A0

#define BNO08X_I2C_ADDR    0x4B
#define VL53_I2C_ADDR      0x29

#define SEND_INTERVAL_MS   33     // 33 ms ≈ 30.3 Hz serial output rate
#define LPF_ALPHA_DISTANCE 0.1f   // low-pass smoothing for the distance sensor
#define LPF_ALPHA_HALL     0.1f   // low-pass smoothing for the hall sensor

// IMU axis sign convention (depends on how the BNO085 is mounted on the tool).
// Matches the original firmware: X and Z inverted, W and Y as-is.
#define CAL_QUAT_SIGN_W    1.0f
#define CAL_QUAT_SIGN_I   -1.0f
#define CAL_QUAT_SIGN_J    1.0f
#define CAL_QUAT_SIGN_K   -1.0f


// ----------------------------------------------------------------------------
//  Per-board profiles
// ----------------------------------------------------------------------------
#if UMI_BOARD_ID == 1
  #define CAL_NAME              "board-1"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        310.0f   // physical usable rod length
  #define CAL_DIST_RAW_MIN_MM   130.0f   // raw reading at the fully-inserted end
  #define CAL_DIST_RAW_MAX_MM   350.0f   // raw reading at the fully-retracted end
  #define CAL_DIST_OUT_MIN_MM   100.0f   // mapped output at CAL_DIST_RAW_MIN_MM

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     2800     // >= this  -> 0 %   (fully open)
  #define CAL_HALL_ADC_25PCT    1930
  #define CAL_HALL_ADC_50PCT    1810
  #define CAL_HALL_ADC_100PCT   1765     // <  this  -> 100 % (fully closed)

#elif UMI_BOARD_ID == 2
  #define CAL_NAME              "board-2"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        315.0f
  #define CAL_DIST_RAW_MIN_MM   116.0f
  #define CAL_DIST_RAW_MAX_MM   310.0f
  #define CAL_DIST_OUT_MIN_MM   100.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     2510
  #define CAL_HALL_ADC_25PCT    2200
  #define CAL_HALL_ADC_50PCT    2142
  #define CAL_HALL_ADC_100PCT   2126


#elif UMI_BOARD_ID == 3
  #define CAL_NAME              "board-3"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        208.0f
  #define CAL_DIST_RAW_MIN_MM   70.0f
  #define CAL_DIST_RAW_MAX_MM   238.0f
  #define CAL_DIST_OUT_MIN_MM   50.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     1978
  #define CAL_HALL_ADC_25PCT    2070
  #define CAL_HALL_ADC_50PCT    2134
  #define CAL_HALL_ADC_100PCT   2178

#elif UMI_BOARD_ID == 4
  #define CAL_NAME              "board-4"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        315.0f
  #define CAL_DIST_RAW_MIN_MM   114.0f
  #define CAL_DIST_RAW_MAX_MM   332.0f
  #define CAL_DIST_OUT_MIN_MM   100.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     1380
  #define CAL_HALL_ADC_25PCT    1808
  #define CAL_HALL_ADC_50PCT    1865
  #define CAL_HALL_ADC_100PCT   1885

#elif UMI_BOARD_ID == 5
  #define CAL_NAME              "board-5"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        285.0f
  #define CAL_DIST_RAW_MIN_MM   104.0f
  #define CAL_DIST_RAW_MAX_MM   301.0f
  #define CAL_DIST_OUT_MIN_MM   100.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     1988
  #define CAL_HALL_ADC_25PCT    1940
  #define CAL_HALL_ADC_50PCT    1930
  #define CAL_HALL_ADC_100PCT   1920


#elif UMI_BOARD_ID == 6
  #define CAL_NAME              "board-6-qc"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        284.0f
  #define CAL_DIST_RAW_MIN_MM   123.0f
  #define CAL_DIST_RAW_MAX_MM   300.0f
  #define CAL_DIST_OUT_MIN_MM   100.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     2060
  #define CAL_HALL_ADC_25PCT    2145
  #define CAL_HALL_ADC_50PCT    2170
  #define CAL_HALL_ADC_100PCT   2180


#elif UMI_BOARD_ID == 7
  #define CAL_NAME              "board-7"

  // --- Insertion depth (VL53L0X), millimetres ---
  #define CAL_ROD_LEN_MM        310.0f
  #define CAL_DIST_RAW_MIN_MM   110.0f
  #define CAL_DIST_RAW_MAX_MM   312.0f
  #define CAL_DIST_OUT_MIN_MM   100.0f

  // --- Gripper (hall sensor) 12-bit ADC breakpoints (0..4095) ---
  #define CAL_HALL_ADC_0PCT     2080
  #define CAL_HALL_ADC_25PCT    1970
  #define CAL_HALL_ADC_50PCT    1920
  #define CAL_HALL_ADC_100PCT   1895

#else
  #error "UMI_BOARD_ID not recognised — add a matching profile block in calibration.h"
#endif


// ----------------------------------------------------------------------------
//  Sanity checks — fail the build early on obviously-bad calibration.
//  NOTE: the C preprocessor #if only evaluates integer arithmetic, so the
//  float distance constants can't be range-checked here — only the integer
//  hall breakpoints are.
//
//  The breakpoints must be strictly MONOTONIC, but either direction is valid:
//  the ADC rises OR falls as the gripper closes depending on the magnet's
//  polarity on that board (boards 1/2 fall, board 3 rises). What we reject is a
//  non-monotonic sequence (a repeated value or a reversal mid-way), which means
//  a mis-measured breakpoint the firmware can't map cleanly.
// ----------------------------------------------------------------------------
#define CAL_HALL_STRICTLY_DECREASING \
  ((CAL_HALL_ADC_0PCT  > CAL_HALL_ADC_25PCT)  && \
   (CAL_HALL_ADC_25PCT > CAL_HALL_ADC_50PCT)  && \
   (CAL_HALL_ADC_50PCT > CAL_HALL_ADC_100PCT))

#define CAL_HALL_STRICTLY_INCREASING \
  ((CAL_HALL_ADC_0PCT  < CAL_HALL_ADC_25PCT)  && \
   (CAL_HALL_ADC_25PCT < CAL_HALL_ADC_50PCT)  && \
   (CAL_HALL_ADC_50PCT < CAL_HALL_ADC_100PCT))

#if !(CAL_HALL_STRICTLY_DECREASING || CAL_HALL_STRICTLY_INCREASING)
  #error "Hall ADC breakpoints must be strictly monotonic across 0/25/50/100 % (all increasing or all decreasing). Direction depends on magnet polarity; a mixed/repeated sequence means a mis-measured breakpoint."
#endif
