// ============================================================================
//  UMI device firmware — dual-stream (processed + raw) sensor output
// ----------------------------------------------------------------------------
//  Stream A (processed, 100Hz): chip-fused rotation vector + MCU-smoothed and
//    calibrated distance/gripper — same role as the legacy 30Hz CSV, now
//    emitted 1:1 on every Rotation Vector report (binary FUSED record).
//  Stream B (raw, new): calibrated accel/gyro/mag straight from the chip
//    (no fusion/smoothing) plus pre-filter ToF/Hall samples, each timestamped,
//    so the host can re-run its own sensor fusion offline.
//
//  Design rules (see MCU/firmware_rewrite_spec.md):
//    * no delay() in loop, fully non-blocking
//    * BNO08x events are drained until empty, gated on the INT pin with a
//      periodic fallback; every report is captured via our own sh2 sensor
//      callback (the stock Adafruit getSensorEvent() keeps only the LAST
//      report of a batched SHTP cargo — a guaranteed loss at 600 reports/s)
//    * VL53L0X is read with the non-blocking readRangeResult() only after
//      isRangeComplete(); the blocking readRange()/rangingTest() paths are
//      never used
//    * serial writes are drop-not-block; every record carries a per-type
//      8-bit sequence number so the host can detect any loss
// ============================================================================

#include <Wire.h>
#include <Adafruit_BNO08x.h>   // also pulls in sh2.h / sh2_SensorValue.h
#include <Adafruit_VL53L0X.h>

// Per-board pin map, sensor addresses, filter and calibration constants.
// Edit calibration.h (not this file) to retune an individual device.
#include "calibration.h"

// Rename the USB device at runtime so the host's serial-port list shows it by
// board name (e.g. "board-2"). Two families are handled:
//   * ESP32 in TinyUSB (USB-OTG) mode — via the Espressif USB.h API. In the
//     default USB-Serial/JTAG mode (and on the C3) the descriptor is fixed in
//     hardware and cannot be renamed.
//   * Adafruit TinyUSB cores (XIAO nRF52840, SAMD21, RP2040) — via
//     TinyUSBDevice. Here the descriptor is fully software-defined, so the
//     rename always takes effect.
// Either way, the Python tools also read the name via the "?" serial handshake
// below, so they display the board name even when the descriptor can't change.
#if defined(ARDUINO_ARCH_ESP32) && defined(ARDUINO_USB_MODE) && (ARDUINO_USB_MODE == 0)
#include "USB.h"
#elif defined(USE_TINYUSB)
#include "Adafruit_TinyUSB.h"
#endif

// ----------------------------------------------------------------------------
//  Record types & framing:  [0xAA][0x55][type][len][seq][payload][crc8]
//  crc8 covers type..payload, polynomial 0x07, init 0x00. Little-endian.
// ----------------------------------------------------------------------------
const uint8_t SYNC0 = 0xAA;
const uint8_t SYNC1 = 0x55;

const uint8_t REC_FUSED    = 0x01;  // 100Hz, triggered by RV arrival
const uint8_t REC_ACCEL    = 0x10;  // 200Hz calibrated accelerometer
const uint8_t REC_GYRO     = 0x11;  // 200Hz calibrated gyroscope
const uint8_t REC_MAG      = 0x12;  // 100Hz calibrated magnetometer
const uint8_t REC_RV       = 0x13;  // 100Hz chip-original rotation vector
const uint8_t REC_TOF      = 0x20;  //  50Hz raw ToF range
const uint8_t REC_HALL     = 0x21;  // 200Hz raw hall ADC
const uint8_t REC_TIMESYNC = 0x30;  //   1Hz chip<->MCU timebase sample
const uint8_t REC_STATS    = 0x31;  //   1Hz health counters
const uint8_t REC_INFO     = 0x7F;  // CAL_NAME string (handshake reply)

// Dense per-type index for seq counters / generation counters.
enum RecIdx {
  IDX_FUSED, IDX_ACCEL, IDX_GYRO, IDX_MAG, IDX_RV, IDX_TOF, IDX_HALL,
  IDX_TIMESYNC, IDX_STATS, IDX_INFO, REC_IDX_COUNT
};
const uint8_t STATS_COUNTED_TYPES = 7;  // IDX_FUSED..IDX_HALL go into STATS

// --- Sensor scheduling (fixed spec values — do not retune) ---
const uint32_t RV_INTERVAL_US       = 10000;   // 100Hz rotation vector
const uint32_t ACCEL_INTERVAL_US    = 5000;    // 200Hz calibrated accel
const uint32_t GYRO_INTERVAL_US     = 5000;    // 200Hz calibrated gyro
const uint32_t MAG_INTERVAL_US      = 10000;   // 100Hz calibrated mag
const uint32_t TOF_TIMING_BUDGET_US = 20000;   // -> 50Hz continuous ranging
const uint32_t IMU_SILENT_RST_SECS  = 2;       // hub silent this long -> hard reset it
const uint16_t TOF_PERIOD_MS        = 1;       // end-to-start gap: 18.5ms meas + 1ms = 50Hz
const uint32_t HALL_PERIOD_US       = 5000;    // 200Hz hall ADC sampling
const uint32_t DRAIN_FALLBACK_US    = 2000;    // drain at least every 2ms even without INT
const uint32_t ONE_HZ_PERIOD_US     = 1000000; // TIMESYNC / STATS cadence
const uint32_t RV_TIMEOUT_US        = 500000;  // RV silence considered an outage
const uint32_t TOF_POLL_US          = 1000;    // isRangeComplete() poll spacing
const uint32_t TOF_QUIET_US         = 15000;   // no polling right after a read (data is ~20ms away)
const uint16_t TOF_VALID_MAX_MM     = 8000;    // >= this is out-of-range, don't feed the LPF
// Pass budget per drain call. Large enough to actually reach an empty queue
// (escaping the accumulated-backlog regime where every cargo read is a big,
// slow multi-chunk transfer), small enough to bound loop latency; the ≤2ms
// fallback brings us straight back for any remainder.
const int      MAX_DRAIN_PASSES     = 32;
const uint8_t  MAX_PAYLOAD_LEN      = 64;

#define UMI_DEBUG 0   // 1 = emit a 1Hz DBG INFO record with scheduler counters

Adafruit_BNO08x bno08x;
Adafruit_VL53L0X vl53 = Adafruit_VL53L0X();

// --- Filter state (identical semantics to the legacy firmware) ---
float filteredDistance = -1.0f;
float filteredAnalog   = -1.0f;

// --- Stream/health counters (STATS record) ---
uint8_t  seqCtr[REC_IDX_COUNT]   = {0};
uint16_t genCount[REC_IDX_COUNT] = {0};  // generated since last STATS (incl. dropped)
uint16_t serialDropCount = 0;            // cumulative, wraps at 65536
uint16_t i2cErrCount     = 0;            // cumulative: decode/enable failures + RV outages
uint8_t  imuResetCount   = 0;            // cumulative BNO08x resets seen

// --- IMU event plumbing ---
volatile bool imuIntFlag = false;  // set by ISR, consumed by loop
bool     imuEventSeen  = false;    // set by the sh2 callback during a service pass
uint32_t imuAnchorUs   = 0;        // micros() taken right before each sh2_service()
uint32_t lastImuEvtUs  = 0;        // reconstructed event time of the latest IMU report
uint32_t lastImuRxMcuUs = 0;       // micros() when that report was received
uint32_t lastRvMcuUs   = 0;
bool     rvActive      = false;    // false until first RV, or after a timeout
uint16_t imuReenableCount = 0;     // self-heal re-enables (no IMU events for >1s)
uint8_t  imuSilentSecs = 0;        // consecutive seconds without any IMU event
uint32_t svcCount      = 0;        // sh2_service() invocations (debug)
uint32_t loopCount     = 0;        // loop() iterations (debug)
uint32_t drainCalls    = 0;        // drainImu() invocations (debug)
uint32_t emptyExits    = 0;        // drains that reached an empty queue (debug)
uint32_t evtCount      = 0;        // IMU events decoded (debug)
uint32_t svcUs         = 0;        // µs spent inside sh2_service (debug)
uint32_t wrUs          = 0;        // µs spent inside Serial.write (debug)

// --- Payload layouts (packed, little-endian on this MCU) ---
struct __attribute__((packed)) FusedPayload {
  uint32_t mcu_us; float qw, qx, qy, qz; float dist_mm; uint8_t grip_pct;
};
// evt_us: sample time on the MCU micros() timebase, reconstructed from the
// hub's per-report delay (see imuSensorHandler) — same clock as ToF/Hall.
struct __attribute__((packed)) ImuVecPayload {
  uint32_t evt_us; float x, y, z; uint8_t status;
};
struct __attribute__((packed)) RvPayload {
  uint32_t evt_us; float qw, qx, qy, qz, acc_rad; uint8_t status;
};
struct __attribute__((packed)) TofPayload {
  uint32_t mcu_us; uint16_t range_mm; uint8_t range_status;
};
struct __attribute__((packed)) HallPayload {
  uint32_t mcu_us; uint16_t adc;
};
struct __attribute__((packed)) TimesyncPayload {
  uint32_t mcu_now_us;      // micros() when this record was built
  uint32_t evt_last_us;     // reconstructed event time of the latest IMU report
  uint32_t mcu_at_rx_us;    // micros() when that report was received
};
struct __attribute__((packed)) StatsPayload {
  uint16_t serial_drop; uint16_t i2c_err; uint8_t imu_reset;
  uint16_t gen[STATS_COUNTED_TYPES];  // FUSED,ACCEL,GYRO,MAG,RV,TOF,HALL since last STATS
};

static_assert(sizeof(FusedPayload)    == 25, "FUSED payload must be 25B");
static_assert(sizeof(ImuVecPayload)   == 17, "ACCEL/GYRO/MAG payload must be 17B");
static_assert(sizeof(RvPayload)       == 25, "RV payload must be 25B");
static_assert(sizeof(TofPayload)      == 7,  "TOF payload must be 7B");
static_assert(sizeof(HallPayload)     == 6,  "HALL payload must be 6B");
static_assert(sizeof(TimesyncPayload) == 12, "TIMESYNC payload must be 12B");
static_assert(sizeof(StatsPayload)    == 19, "STATS payload must be 19B");

// --- 홀센서 ADC -> 그리퍼 개폐율(%) 매핑 ---
// Piecewise-linear map from a raw hall ADC reading to gripper closure %.
// The four calibration breakpoints are strictly monotonic in ADC, but the
// DIRECTION depends on the magnet's polarity: the reading may rise OR fall as
// the gripper closes (boards 1/2 fall, board 3 rises — see calibration.h).
// This routine works for both directions without any per-board branching:
// each segment is bracketed by min/max of its endpoints, and map() interpolates
// correctly whether the ADC endpoints ascend or descend. Values outside the
// calibrated span are clamped to 0 % / 100 %.
int hallToPercent(int adc) {
  const int adcBp[4] = { CAL_HALL_ADC_0PCT, CAL_HALL_ADC_25PCT,
                         CAL_HALL_ADC_50PCT, CAL_HALL_ADC_100PCT };
  const int pctBp[4] = { 0, 25, 50, 100 };

  for (int i = 0; i < 3; i++) {
    int aLo = min(adcBp[i], adcBp[i + 1]);
    int aHi = max(adcBp[i], adcBp[i + 1]);
    if (adc >= aLo && adc <= aHi) {
      return map(adc, adcBp[i], adcBp[i + 1], pctBp[i], pctBp[i + 1]);
    }
  }

  // Outside the calibrated span — clamp to the nearer end.
  // "decreasing" == ADC falls as the gripper closes (0 % end has the higher ADC).
  bool decreasing = adcBp[0] > adcBp[3];
  if (decreasing) return (adc > adcBp[0]) ? 0 : 100;
  else            return (adc < adcBp[0]) ? 0 : 100;
}

// Map the LPF-smoothed raw ToF distance to the calibrated insertion depth (mm)
// — identical arithmetic to the legacy firmware's 30Hz output block.
float mapDistanceMm() {
  const float rod_len = CAL_ROD_LEN_MM;
  float finalDistance = 0.0f;  // error / not-yet-initialised default
  if (filteredDistance > 0) {
    if (filteredDistance <= CAL_DIST_RAW_MIN_MM) {
      finalDistance = CAL_DIST_OUT_MIN_MM;
    } else if (filteredDistance >= CAL_DIST_RAW_MAX_MM) {
      finalDistance = rod_len;
    } else {
      finalDistance = CAL_DIST_OUT_MIN_MM + (filteredDistance - CAL_DIST_RAW_MIN_MM) *
                      ((rod_len - CAL_DIST_OUT_MIN_MM) / (CAL_DIST_RAW_MAX_MM - CAL_DIST_RAW_MIN_MM));
    }
  }
  return rod_len - finalDistance;
}

// ----------------------------------------------------------------------------
//  Framing / transmission
// ----------------------------------------------------------------------------
uint8_t crc8(const uint8_t *data, size_t len) {
  uint8_t crc = 0x00;
  while (len--) {
    crc ^= *data++;
    for (uint8_t bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

int8_t recIndex(uint8_t type) {
  switch (type) {
    case REC_FUSED:    return IDX_FUSED;
    case REC_ACCEL:    return IDX_ACCEL;
    case REC_GYRO:     return IDX_GYRO;
    case REC_MAG:      return IDX_MAG;
    case REC_RV:       return IDX_RV;
    case REC_TOF:      return IDX_TOF;
    case REC_HALL:     return IDX_HALL;
    case REC_TIMESYNC: return IDX_TIMESYNC;
    case REC_STATS:    return IDX_STATS;
    case REC_INFO:     return IDX_INFO;
    default:           return -1;
  }
}

// Single transmit path: framing + per-type seq + CRC + drop-not-block.
// The seq counter advances even when the frame is dropped, so the host sees
// a gap (loss detection) while serial_drop tells it the drop was local.
void sendRecord(uint8_t type, const void *payload, uint8_t len) {
#if LEGACY_CSV
  // Legacy mode outputs only the 7-field CSV (printed in emitFused()).
  (void)type; (void)payload; (void)len;
  return;
#else
  int8_t idx = recIndex(type);
  if (idx < 0 || len > MAX_PAYLOAD_LEN) return;

  uint8_t seq = seqCtr[idx]++;
  genCount[idx]++;

  uint8_t frame[5 + MAX_PAYLOAD_LEN + 1];
  frame[0] = SYNC0;
  frame[1] = SYNC1;
  frame[2] = type;
  frame[3] = len;
  frame[4] = seq;
  memcpy(&frame[5], payload, len);
  frame[5 + len] = crc8(&frame[2], (size_t)len + 3);

  size_t total = (size_t)len + 6;
  if ((size_t)Serial.availableForWrite() < total) {
    serialDropCount++;  // never block: a stalled I2C drain costs more than one frame
    return;
  }
  uint32_t w0 = micros();
  Serial.write(frame, total);
  wrUs += micros() - w0;
#endif
}

// ----------------------------------------------------------------------------
//  IMU: ISR + sh2 sensor callback + drain
// ----------------------------------------------------------------------------
void imuIsr() {
  imuIntFlag = true;  // flag only — no I2C/Serial in ISR context
}

// FUSED record: triggered by RV arrival (never by a timer), so RV:FUSED is 1:1.
void emitFused(const sh2_SensorValue_t &v, uint32_t rxUs) {
  float qw = CAL_QUAT_SIGN_W * v.un.rotationVector.real;
  float qx = CAL_QUAT_SIGN_I * v.un.rotationVector.i;
  float qy = CAL_QUAT_SIGN_J * v.un.rotationVector.j;
  float qz = CAL_QUAT_SIGN_K * v.un.rotationVector.k;
  float distMm = mapDistanceMm();
  int grip = hallToPercent((int)filteredAnalog);
  if (grip < 0) grip = 0;
  if (grip > 100) grip = 100;

#if LEGACY_CSV
  // Legacy 7-field CSV, now at 100Hz (the host CSV parser is rate-agnostic).
  Serial.print(rxUs / 1000); Serial.print(",");
  Serial.print(qw, 4);       Serial.print(",");
  Serial.print(qx, 4);       Serial.print(",");
  Serial.print(qy, 4);       Serial.print(",");
  Serial.print(qz, 4);       Serial.print(",");
  Serial.print(distMm, 2);   Serial.print(",");
  Serial.println(grip);
#else
  FusedPayload p;
  p.mcu_us   = rxUs;
  p.qw = qw; p.qx = qx; p.qy = qy; p.qz = qz;
  p.dist_mm  = distMm;
  p.grip_pct = (uint8_t)grip;
  sendRecord(REC_FUSED, &p, sizeof(p));
#endif
}

// Registered with sh2_setSensorCallback() *instead of* the Adafruit handler:
// the stock handler copies each decoded event into one shared value, so when
// an SHTP cargo carries several batched reports only the last one survives a
// getSensorEvent() call. Handling every event here is what makes the raw
// streams lossless. Runs in sh2_service() context (normal code, not ISR).
void imuSensorHandler(void *cookie, sh2_SensorEvent_t *event) {
  (void)cookie;
  sh2_SensorValue_t v;
  if (sh2_decodeSensorEvent(&v, event) != SH2_OK) {
    i2cErrCount++;
    return;
  }
  imuEventSeen = true;
  evtCount++;

  // Event-time reconstruction. The Adafruit HAL hands sh2 a read timestamp of
  // 0, so v.timestamp is not a chip timebase — it is the (negative) delay
  // from the physical sample to the SHTP read, e.g. ~-3000µs, wrapped into
  // uint32. Anchoring that delay to micros() captured just before the
  // sh2_service() read puts every IMU sample on the SAME MCU µs timebase as
  // the ToF/Hall/FUSED records — no cross-timebase regression needed offline.
  uint32_t rxUs   = micros();
  int32_t  delta  = (int32_t)(uint32_t)v.timestamp;
  // Sanity-clamp: the delay must be small and non-positive. The sh2 timebase
  // decode occasionally produces a ~1s bogus value (32kHz reference rollover
  // with a zero read-timestamp); fall back to the anchor time in that case.
  if (delta > 1000 || delta < -100000) delta = 0;
  uint32_t evtUs = imuAnchorUs + (uint32_t)delta;
  lastImuEvtUs   = evtUs;
  lastImuRxMcuUs = rxUs;

  switch (v.sensorId) {
    case SH2_ACCELEROMETER: {
      ImuVecPayload p = { evtUs, v.un.accelerometer.x, v.un.accelerometer.y,
                          v.un.accelerometer.z, v.status };
      sendRecord(REC_ACCEL, &p, sizeof(p));
      break;
    }
    case SH2_GYROSCOPE_CALIBRATED: {
      ImuVecPayload p = { evtUs, v.un.gyroscope.x, v.un.gyroscope.y,
                          v.un.gyroscope.z, v.status };
      sendRecord(REC_GYRO, &p, sizeof(p));
      break;
    }
    case SH2_MAGNETIC_FIELD_CALIBRATED: {
      ImuVecPayload p = { evtUs, v.un.magneticField.x, v.un.magneticField.y,
                          v.un.magneticField.z, v.status };
      sendRecord(REC_MAG, &p, sizeof(p));
      break;
    }
    case SH2_ROTATION_VECTOR: {
      // 0x13 carries the chip-original quaternion (no sign remap) — it is the
      // reference for offline fusion. The remap is applied only in FUSED.
      RvPayload p = { evtUs,
                      v.un.rotationVector.real, v.un.rotationVector.i,
                      v.un.rotationVector.j,    v.un.rotationVector.k,
                      v.un.rotationVector.accuracy, v.status };
      sendRecord(REC_RV, &p, sizeof(p));
      emitFused(v, rxUs);
      lastRvMcuUs = rxUs;
      rvActive = true;
      break;
    }
    default:
      break;
  }
}

// Service the SHTP queue until it is empty (bounded to keep the loop alive).
void drainImu() {
  drainCalls++;
  for (int pass = 0; pass < MAX_DRAIN_PASSES; pass++) {
    imuEventSeen = false;
    svcCount++;
    uint32_t s0 = micros();
    imuAnchorUs = s0;       // timebase anchor for this pass's event timestamps
    sh2_service();          // dispatches every batched report to the callback
    svcUs += micros() - s0;
    if (!imuEventSeen) { emptyExits++; break; }
  }
}

// ----------------------------------------------------------------------------
//  VL53L0X direct register helpers
// ----------------------------------------------------------------------------
const uint8_t VL53_REG_SYSRANGE_START     = 0x00;
const uint8_t VL53_REG_STOP_VARIABLE      = 0x91;
const uint8_t VL53_REG_INTERRUPT_STATUS   = 0x13;
const uint8_t VL53_INT_STATUS_MASK        = 0x07;
const uint8_t VL53_REG_SOFT_RESET         = 0xBF;

uint8_t vl53ReadReg(uint8_t reg) {
  Wire.beginTransmission(VL53_I2C_ADDR);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom((uint8_t)VL53_I2C_ADDR, (uint8_t)1);
  return Wire.read();
}

void vl53WriteReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(VL53_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// Data-ready check via the interrupt status register. The wrapper's
// isRangeComplete() branches on its internal DeviceMode (left at SINGLE since
// we start ranging ourselves) and would read the wrong register.
bool vl53DataReady() {
  return (vl53ReadReg(VL53_REG_INTERRUPT_STATUS) & VL53_INT_STATUS_MASK) != 0;
}

// Full software reset (ST's VL53L0X_ResetDevice register sequence): returns
// the device to its power-on state no matter what the previous firmware
// session left running (a DFU upload reboots only the MCU — the VL53L0X
// keeps ranging, which makes a plain re-init flaky).
void vl53SoftReset() {
  vl53WriteReg(VL53_REG_SOFT_RESET, 0x00);
  delay(5);
  vl53WriteReg(VL53_REG_SOFT_RESET, 0x01);
  delay(10);  // t_boot is ~1.2ms; leave margin
}

// Enable all reports in one place — used by setup() AND reset recovery, so a
// BNO08x reset restores every stream (the legacy firmware only restored RV).
bool enableImuReports() {
  bool ok = true;
  ok &= bno08x.enableReport(SH2_ROTATION_VECTOR, RV_INTERVAL_US);
#if !LEGACY_CSV
  ok &= bno08x.enableReport(SH2_ACCELEROMETER, ACCEL_INTERVAL_US);
  ok &= bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, GYRO_INTERVAL_US);
  ok &= bno08x.enableReport(SH2_MAGNETIC_FIELD_CALIBRATED, MAG_INTERVAL_US);
#endif
  if (!ok) i2cErrCount++;
  return ok;
}

// ----------------------------------------------------------------------------
//  Hardware watchdog — last line of defence against deep hangs
// ----------------------------------------------------------------------------
// A ~30min soak test produced a hang so deep even the USB DFU touch stopped
// responding (suspected TWIM bus lockup or hard fault — unrecoverable from
// software). The WDT runs independently of the CPU: if loop() stops feeding
// it for WDT_TIMEOUT_S, the chip reboots and setup()'s recovery path (bus
// clear, begin retries, sensor resets) brings everything back automatically.
const uint32_t WDT_TIMEOUT_S = 5;
bool bootedFromWatchdog = false;

void startWatchdog() {
  // Note which reset brought us up, then clear the latch for next time.
  bootedFromWatchdog = (NRF_POWER->RESETREAS & POWER_RESETREAS_DOG_Msk) != 0;
  NRF_POWER->RESETREAS = 0xFFFFFFFF;

  NRF_WDT->CONFIG = (WDT_CONFIG_HALT_Pause << WDT_CONFIG_HALT_Pos) |
                    (WDT_CONFIG_SLEEP_Run << WDT_CONFIG_SLEEP_Pos);
  NRF_WDT->CRV = WDT_TIMEOUT_S * 32768;
  NRF_WDT->RREN = WDT_RREN_RR0_Msk;
  NRF_WDT->TASKS_START = 1;
}

inline void feedWatchdog() {
  NRF_WDT->RR[0] = WDT_RR_RR_Reload;
}

// Free a stuck I2C bus before Wire.begin(): if a slave was left mid-read by
// the previous session (DFU reboots only the MCU), it can hold SDA low until
// it gets clock pulses — the classic "works after unplug/replug" failure.
// Standard bus-clear: up to 16 SCL pulses until SDA releases, then a STOP.
void i2cBusClear() {
  pinMode(PIN_WIRE_SDA, INPUT_PULLUP);
  pinMode(PIN_WIRE_SCL, INPUT_PULLUP);
  delayMicroseconds(10);
  if (digitalRead(PIN_WIRE_SDA) == HIGH) return;  // bus is fine

  pinMode(PIN_WIRE_SCL, OUTPUT);
  for (int i = 0; i < 16 && digitalRead(PIN_WIRE_SDA) == LOW; i++) {
    digitalWrite(PIN_WIRE_SCL, LOW);
    delayMicroseconds(5);
    digitalWrite(PIN_WIRE_SCL, HIGH);
    delayMicroseconds(5);
  }
  // Generate a STOP (SDA low->high while SCL high) to leave the bus idle.
  pinMode(PIN_WIRE_SDA, OUTPUT);
  digitalWrite(PIN_WIRE_SDA, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_WIRE_SDA, HIGH);
  delayMicroseconds(5);
  pinMode(PIN_WIRE_SDA, INPUT_PULLUP);
  pinMode(PIN_WIRE_SCL, INPUT_PULLUP);
}

// ----------------------------------------------------------------------------
//  Setup
// ----------------------------------------------------------------------------
void setup() {
  // Name the USB device after the active calibration profile so it appears as
  // e.g. "board-2" in the host's serial-port list.
#if defined(ARDUINO_ARCH_ESP32) && defined(ARDUINO_USB_MODE) && (ARDUINO_USB_MODE == 0)
  // ESP32 (TinyUSB/USB-OTG): set names before starting the USB stack.
  USB.productName(CAL_NAME);
  USB.manufacturerName("SurgiTag UMI");
  USB.serialNumber(CAL_NAME);
  USB.begin();
#elif defined(USE_TINYUSB)
  // Adafruit TinyUSB (XIAO nRF52840, SAMD21, RP2040): set the descriptor
  // strings, and if USB is already enumerated, briefly detach so the host
  // re-reads them and picks up the new product name.
  TinyUSBDevice.setManufacturerDescriptor("SurgiTag UMI");
  TinyUSBDevice.setProductDescriptor(CAL_NAME);
  if (TinyUSBDevice.mounted()) {
    TinyUSBDevice.detach();
    delay(10);
    TinyUSBDevice.attach();
  }
#endif

  Serial.begin(115200);
  // Don't gate on the host opening the port: after a watchdog recovery mid-
  // episode there may be no DTR toggle, and the device must resume streaming
  // on its own. Give USB a short window to come up, then proceed regardless.
  for (int i = 0; i < 200 && !Serial; i++) delay(10);

  startWatchdog();   // from here on, a >5s stall reboots us into recovery

  i2cBusClear();
  Wire.begin();
  Wire.setClock(400000);

  // BNO085 bring-up: hard reset (RST = D10) then begin, retried — right after
  // a DFU upload the hub can be mid-dump of a queue nobody was reading and
  // miss the first begin (the old firmware hung forever here).
  pinMode(BNO08X_RST_PIN, OUTPUT);
  digitalWrite(BNO08X_RST_PIN, HIGH);
  bool imuUp = false;
  for (int attempt = 0; attempt < 10 && !imuUp; attempt++) {
    feedWatchdog();   // a slow-but-succeeding bring-up must not trip the WDT
    delay(10);
    digitalWrite(BNO08X_RST_PIN, LOW);
    delay(10);
    digitalWrite(BNO08X_RST_PIN, HIGH);
    delay(150);
    imuUp = bno08x.begin_I2C(BNO08X_I2C_ADDR, &Wire, 0);
    if (!imuUp) {  // field diagnostics: does anyone ACK the bus?
      Wire.beginTransmission(BNO08X_I2C_ADDR);
      int eBno = Wire.endTransmission();
      Wire.beginTransmission(VL53_I2C_ADDR);
      int eVl = Wire.endTransmission();
      char m[64];
      snprintf(m, sizeof(m), "begin fail #%d bno_probe=%d vl53_probe=%d", attempt, eBno, eVl);
      Serial.println(m);
    }
  }
  if (!imuUp) {
    Serial.println("Error: BNO085 Failed.");
    while (1) delay(10);
  }
  // Take over sensor-event delivery from the Adafruit single-value handler so
  // batched reports are never lost (see imuSensorHandler above). The async
  // event path (wasReset) is a separate callback and stays with the library.
  sh2_setSensorCallback(imuSensorHandler, NULL);

#if !UMI_IMU_ONLY
  // VL53L0X bring-up: soft-reset to power-on state first (it may still be
  // ranging from a previous firmware session), then init — retried like the BNO.
  bool tofUp = false;
  for (int attempt = 0; attempt < 5 && !tofUp; attempt++) {
    feedWatchdog();
    vl53SoftReset();
    tofUp = vl53.begin(VL53_I2C_ADDR, false, &Wire);
    if (!tofUp) {  // field diagnostics
      Wire.beginTransmission(VL53_I2C_ADDR);
      int eVl = Wire.endTransmission();
      char m[48];
      snprintf(m, sizeof(m), "vl53 fail #%d probe=%d", attempt, eVl);
      Serial.println(m);
    }
  }
  if (!tofUp) {
    Serial.println("Error: VL53L0X Failed.");
    while (1) delay(10);
  }
  vl53.setMeasurementTimingBudgetMicroSeconds(TOF_TIMING_BUDGET_US);
  // Timed continuous with a 1ms inter-measurement gap. The device's real
  // measurement time under a 20ms budget is ~18.5ms and the timed period is
  // an END-to-start delay, so: back-to-back ran at ~54Hz (out of the ±5%
  // window) and 20ms period gave ~26Hz; 18.5 + ~1ms lands on the 50Hz target.
  vl53.startRangeContinuous(TOF_PERIOD_MS);
#endif  // !UMI_IMU_ONLY

#if !UMI_IMU_ONLY
  pinMode(HALL_SENSOR_PIN, INPUT);
  analogReadResolution(12);
#endif  // !UMI_IMU_ONLY

  // Re-assert the bus speed AFTER every sensor begin: the Adafruit init paths
  // call Wire.begin() again internally, which resets the clock to the 100kHz
  // default — leaving it there triples every cargo read and saturates the loop.
  Wire.setClock(400000);

  // INT (D9) falls when the BNO08x has data; the ISR only raises a flag.
  pinMode(BNO08X_INT_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BNO08X_INT_PIN), imuIsr, FALLING);

  // Start the IMU streams LAST. At 600 reports/s the hub fills a >384B SHTP
  // cargo in ~35ms, and the Adafruit HAL can never read a cargo bigger than
  // its buffer (it returns 0 without consuming — a permanent stall), so the
  // streams must not run while setup is still doing slow I2C work. A fresh
  // RST pulse here puts the hub through the one startup path that reliably
  // produces data: reset event -> wasReset() -> enableImuReports() in loop(),
  // with the ≤2ms drain cadence keeping every cargo small from then on.
  enableImuReports();
  drainImu();
  digitalWrite(BNO08X_RST_PIN, LOW);
  delayMicroseconds(500);
  digitalWrite(BNO08X_RST_PIN, HIGH);

#if LEGACY_CSV
  Serial.print("Initialization complete. Calibration profile: ");
  Serial.println(CAL_NAME);
  if (bootedFromWatchdog) Serial.println("WARN: recovered by watchdog reset");
#else
  sendRecord(REC_INFO, CAL_NAME, (uint8_t)strlen(CAL_NAME));
  if (bootedFromWatchdog) {
    const char *msg = "BOOT watchdog-reset";
    sendRecord(REC_INFO, msg, (uint8_t)strlen(msg));
  }
#endif
}

// ----------------------------------------------------------------------------
//  Main loop — fully non-blocking, micros()-based software timers only
// ----------------------------------------------------------------------------
uint32_t lastDrainUs   = 0;
uint32_t nextTofPollUs = 0;
uint32_t nextHallUs    = 0;
uint32_t nextOneHzUs   = 0;
bool     timersPrimed  = false;

void loop() {
  feedWatchdog();
  loopCount++;
  uint32_t now = micros();
  if (!timersPrimed) {
    timersPrimed  = true;
    lastDrainUs   = now;
    nextTofPollUs = now;
    nextHallUs    = now + HALL_PERIOD_US;
    nextOneHzUs   = now + ONE_HZ_PERIOD_US;
  }

  // --- Host handshake: '?' -> board name (INFO record / legacy ASCII) ---
  while (Serial.available()) {
    if (Serial.read() == '?') {
#if LEGACY_CSV
      Serial.print("NAME,");
      Serial.println(CAL_NAME);
#else
      sendRecord(REC_INFO, CAL_NAME, (uint8_t)strlen(CAL_NAME));
#endif
    }
  }

  // --- IMU drain: INT-gated with a periodic fallback (never trust INT alone) ---
  if (imuIntFlag || (int32_t)(now - lastDrainUs) >= (int32_t)DRAIN_FALLBACK_US) {
    imuIntFlag = false;
    drainImu();
    lastDrainUs = micros();
  }

  // --- BNO08x reset recovery: re-enable EVERY report, count the event ---
  if (bno08x.wasReset()) {
    imuResetCount++;
    enableImuReports();
  }

#if !UMI_IMU_ONLY
  // --- ToF: poll only when data can be ready, read without blocking ---
  now = micros();
  if ((int32_t)(now - nextTofPollUs) >= 0) {
    if (vl53DataReady()) {
      uint32_t sampleUs = micros();
      uint16_t rangeMm  = vl53.readRangeResult();  // non-blocking result fetch
      uint8_t  rStatus  = vl53.readRangeStatus();

      TofPayload p = { sampleUs, rangeMm, rStatus };
      sendRecord(REC_TOF, &p, sizeof(p));

      // LPF on valid samples only — same semantics as the legacy firmware.
      if (rangeMm < TOF_VALID_MAX_MM) {
        float raw = (float)rangeMm;
        if (filteredDistance < 0) filteredDistance = raw;
        else filteredDistance = (LPF_ALPHA_DISTANCE * raw) +
                                ((1.0f - LPF_ALPHA_DISTANCE) * filteredDistance);
      }

      // Next result is a full timing budget away — stay off the bus until then,
      // and give the IMU the bus right after this VL53 transaction burst.
      nextTofPollUs = micros() + TOF_QUIET_US;
      drainImu();
      lastDrainUs = micros();
    } else {
      nextTofPollUs = now + TOF_POLL_US;
    }
  }

  // --- Hall: 200Hz software timer, raw record + LPF update ---
  now = micros();
  if ((int32_t)(now - nextHallUs) >= 0) {
    uint32_t sampleUs = micros();
    uint16_t adc = (uint16_t)analogRead(HALL_SENSOR_PIN);

    HallPayload p = { sampleUs, adc };
    sendRecord(REC_HALL, &p, sizeof(p));

    if (filteredAnalog < 0) filteredAnalog = (float)adc;
    else filteredAnalog = (LPF_ALPHA_HALL * (float)adc) +
                          ((1.0f - LPF_ALPHA_HALL) * filteredAnalog);

    nextHallUs += HALL_PERIOD_US;  // drift-free cadence
    if ((int32_t)(micros() - nextHallUs) >= (int32_t)HALL_PERIOD_US) {
      nextHallUs = micros() + HALL_PERIOD_US;  // fell far behind — resync
    }
  }

#endif  // !UMI_IMU_ONLY

  // --- 1Hz: TIMESYNC + STATS + RV outage detection ---
  now = micros();
  if ((int32_t)(now - nextOneHzUs) >= 0) {
    nextOneHzUs += ONE_HZ_PERIOD_US;
    if ((int32_t)(micros() - nextOneHzUs) >= (int32_t)ONE_HZ_PERIOD_US) {
      nextOneHzUs = micros() + ONE_HZ_PERIOD_US;
    }

    // RV outage: FUSED stops by itself (it is RV-triggered); count the event.
    if (rvActive && (int32_t)(micros() - lastRvMcuUs) >= (int32_t)RV_TIMEOUT_US) {
      rvActive = false;
      i2cErrCount++;  // reported via the STATS i2c_err field
    }

    // Self-heal: if the hub has been silent for a full second, re-enable the
    // reports; if that doesn't bring it back (e.g. an oversized SHTP cargo has
    // wedged the read path), hard-reset the hub via RST. The reset event then
    // flows through wasReset() -> enableImuReports() like any other reset.
    if ((int32_t)(micros() - lastImuRxMcuUs) >= (int32_t)ONE_HZ_PERIOD_US) {
      imuReenableCount++;
      if (++imuSilentSecs >= IMU_SILENT_RST_SECS) {
        imuSilentSecs = 0;
        digitalWrite(BNO08X_RST_PIN, LOW);
        delayMicroseconds(500);  // RST min low time is ns-scale; 500µs is safe
        digitalWrite(BNO08X_RST_PIN, HIGH);
      } else {
        enableImuReports();
      }
    } else {
      imuSilentSecs = 0;
    }

#if UMI_DEBUG
    {
      char dbg[80];
      snprintf(dbg, sizeof(dbg), "DBG i%d L%lu D%lu S%lu E%lu X%lu sv%lu wr%lu",
               (int)digitalRead(BNO08X_INT_PIN), (unsigned long)loopCount,
               (unsigned long)drainCalls, (unsigned long)svcCount,
               (unsigned long)evtCount, (unsigned long)emptyExits,
               (unsigned long)(svcUs / 1000), (unsigned long)(wrUs / 1000));
      sendRecord(REC_INFO, dbg, (uint8_t)strlen(dbg));
      loopCount = drainCalls = svcCount = evtCount = emptyExits = 0;
      svcUs = wrUs = 0;
    }
#endif

    TimesyncPayload ts = { micros(), lastImuEvtUs, lastImuRxMcuUs };
    sendRecord(REC_TIMESYNC, &ts, sizeof(ts));

    StatsPayload st;
    st.serial_drop = serialDropCount;
    st.i2c_err     = i2cErrCount;
    st.imu_reset   = imuResetCount;
    for (uint8_t i = 0; i < STATS_COUNTED_TYPES; i++) {
      st.gen[i] = genCount[i];
      genCount[i] = 0;  // gen[] = records generated in the last second
    }
    sendRecord(REC_STATS, &st, sizeof(st));
  }
}
