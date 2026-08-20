// I2C bus scan + IMU identification for the XIAO nRF52840 (Sense).
// Prints every ACKing address on both the external Wire bus and, on the Sense,
// the internal Wire1 bus that carries the onboard LSM6DS3TR-C.
#include <Wire.h>

static void scanBus(TwoWire &bus, const char *label) {
  Serial.print("[");
  Serial.print(label);
  Serial.println("] scanning 0x08..0x77");
  int found = 0;
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    bus.beginTransmission(addr);
    if (bus.endTransmission() == 0) {
      found++;
      Serial.print("  found 0x");
      if (addr < 16) Serial.print("0");
      Serial.print(addr, HEX);
      switch (addr) {
        case 0x28: case 0x29: Serial.print("  (BNO055 / VL53L0X)"); break;
        case 0x4A: case 0x4B: Serial.print("  (BNO08x IMU)");       break;
        case 0x6A: case 0x6B: Serial.print("  (LSM6DS3TR-C IMU)");  break;
        case 0x1C: case 0x1E: Serial.print("  (LIS3MDL / mag)");    break;
        case 0x0C: case 0x0D: Serial.print("  (AK09918 mag)");      break;
        case 0x68: case 0x69: Serial.print("  (MPU6050/9250)");     break;
        default: break;
      }
      Serial.println();
    }
  }
  Serial.print("  total: ");
  Serial.println(found);
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 300 && !Serial; i++) delay(10);

  // Power the Sense's internal sensor rail (LSM6DS3TR-C sits on Wire1).
#ifdef PIN_LSM6DS3TR_C_POWER
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(20);
#endif

  Wire.begin();
  Wire.setClock(100000);
  Wire1.begin();
  Wire1.setClock(100000);
}

void loop() {
  Serial.println("==== I2C SCAN ====");
  scanBus(Wire, "Wire  (external D4/D5)");
  scanBus(Wire1, "Wire1 (internal, Sense IMU)");
  Serial.println();
  delay(2000);
}
