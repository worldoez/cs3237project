/*
ESP32 + Adafruit MPU6050
Posts batched JSON samples to FastAPI endpoint /ingest at ~50 Hz sampling, 10-sample batches.
JSON schema:
{ "device_id":"imu01", "samples":[ {"ts":..., "gx":..,"gy":..,"gz":..,"ax":..,"ay":..,"az":..}, ... ] }
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

// WIFI CONFIG: set your WiFi and server
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";
const char* SERVER_URL = "http://<laptop_ip>:5000/ingest"; // replace with your PC IP
const char* DEVICE_ID  = "imu01";

const uint32_t SAMPLE_DELAY_MS = 20; // ~50 Hz
const uint32_t CONNECT_RETRY_MS = 2000;
const int BATCH_N = 3; // post every 3 samples

struct Sample {
  uint32_t ts;
  float gx, gy, gz, ax, ay, az;
};

Sample batch[BATCH_N];
int batch_idx = 0;

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    if (millis() - start > 15000) break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);

  connectWiFi();

  if (!mpu.begin()) {
    Serial.println("IMU not found");
    while (1) delay(100);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  delay(100);
}

void postBatch() {
  if (batch_idx == 0) return;
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi not connected, skip post");
      return;
    }
  }
  String payload;
  payload.reserve(256 + batch_idx * 128);
  payload += "{\"device_id\":\"";
  payload += DEVICE_ID;
  payload += "\",\"samples\":[";
  for (int i = 0; i < batch_idx; ++i) {
    if (i) payload += ",";
    payload += "{";
    payload += "\"ts\":" + String(batch[i].ts);
    payload += ",\"gx\":" + String(batch[i].gx, 6);
    payload += ",\"gy\":" + String(batch[i].gy, 6);
    payload += ",\"gz\":" + String(batch[i].gz, 6);
    payload += ",\"ax\":" + String(batch[i].ax, 6);
    payload += ",\"ay\":" + String(batch[i].ay, 6);
    payload += ",\"az\":" + String(batch[i].az, 6);
    payload += "}";
  }
  payload += "]}";

  HTTPClient http;
  http.begin(SERVER_URL);
  http.setReuse(true); 
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Connection", "keep-alive");
  int code = http.POST((uint8_t*)payload.c_str(), payload.length());
  String resp = http.getString();
  http.end();

  // Debug to Serial (optional)
  Serial.print("POST "); Serial.print(code); Serial.print(" len=");
  Serial.print(batch_idx); Serial.print(" resp=");
  Serial.println(resp);

  batch_idx = 0; // reset
}

void loop() {
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  Sample s;
  s.ts = millis();
  s.gx = gyro.gyro.x;  // rad/s
  s.gy = gyro.gyro.y;
  s.gz = gyro.gyro.z;
  s.ax = accel.acceleration.x; // m/s^2
  s.ay = accel.acceleration.y;
  s.az = accel.acceleration.z;

  batch[batch_idx++] = s;
  if (batch_idx >= BATCH_N) {
    postBatch();
  }

  delay(SAMPLE_DELAY_MS);
}