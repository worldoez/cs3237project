/*
ESP32 + Adafruit MPU6050
WAITS FOR BUTTON PRESS to start data collection.
Posts batched JSON samples to FastAPI endpoint /ingest at ~50 Hz sampling, 3-sample batches.
Enters deep sleep when server detects JUMP command.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

const int BUTTON_PIN = 14;
const int BUZZER_PIN = 26;

// CONFIG: WiFi Credentials and Server Address
// ---------------------------------------------------------------------------------------
const char* WIFI_SSID = "SM-S908W1831";
const char* WIFI_PASS = "fbcw5940";

const char* SERVER_URL = "http://10.205.18.152:5000/ingest";
// ---------------------------------------------------------------------------------------

const int WIFI_RETRY_MS = 2000;

// HTTP configurations
const char* DEVICE_ID = "imu01";
const uint32_t SAMPLE_DELAY_MS = 20;  // ~50 Hz
const int BATCH_N = 3;                // post every 3 samples

// Deep sleep configurations
#define uS_TO_S_FACTOR 1000000ULL
RTC_DATA_ATTR int bootCount = 0;

// State management - Persist across deep sleep
RTC_DATA_ATTR bool dataCollectionStarted = false;

// Struct for an IMU datapoint
struct Sample {
  uint32_t ts;
  float gx, gy, gz, ax, ay, az;
};

// Array to store each IMU datapoints
Sample batch[BATCH_N];
int batch_idx = 0;

// Sound the buzzer count times, each for duration with pause in between
void buzzPattern(int count, int duration = 150, int pause = 150) {
  for (int i = 0; i < count; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < count - 1) delay(pause);
  }
}

void buzzLong(int duration = 500) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(duration);
  digitalWrite(BUZZER_PIN, LOW);
}

// Enter deep sleep
void enterDeepSleep() {
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║       ESP32 IMU Device Sleeping    ║");
  Serial.println("╚════════════════════════════════════╝");

  // Configure wake on button press
  esp_sleep_enable_ext0_wakeup((gpio_num_t)BUTTON_PIN, 0);  // LOW (button pressed)

  Serial.println("Entering deep sleep...");
  Serial.println("Wake trigger: Button press\n");
  Serial.flush();

  esp_deep_sleep_start();
  // Never reaches here
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    if (millis() - start > WIFI_RETRY_MS) break;
  }
}

// Only used for when ESP32 is powered
void waitForButtonPress() {
  // Single short beep to indicate ready
  buzzPattern(1, 100);

  Serial.println("Press button to start...");

  // Wait for button press (button is active LOW with pullup)
  while (digitalRead(BUTTON_PIN) == HIGH) {
    delay(50);
  }

  // Button pressed - wait for release to avoid bouncing
  delay(50);
  while (digitalRead(BUTTON_PIN) == LOW) {
    delay(50);
  }

  // Confirmation beep
  buzzPattern(2, 100, 100);
  Serial.println("Button pressed!");
  Serial.println("Starting...\n");
  dataCollectionStarted = true;
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(BUZZER_PIN, OUTPUT);

  // Print boot count
  bootCount++;
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║       ESP32 IMU Device Booted      ║");
  Serial.println("╚════════════════════════════════════╝");
  Serial.println("Boot count: " + String(bootCount));

  // Check wakeup reason
  esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();
  bool wokeFromSleep = false;

  switch (wakeup_reason) {
    case ESP_SLEEP_WAKEUP_EXT0:
      Serial.println("Woke up from: Button press");
      buzzPattern(2);  // Double beep on wake
      wokeFromSleep = true;
      dataCollectionStarted = true;  // Auto-start on wake
      break;
    case ESP_SLEEP_WAKEUP_TIMER:
      Serial.println("Woke up from: Timer");
      buzzPattern(1);  // Single beep on timer wake
      wokeFromSleep = true;
      dataCollectionStarted = true;  // Auto-start on wake
      break;
    default:
      Serial.println("Woke up from: Power on/Reset");
      break;
  }

  Serial.println();  // Print "\n"

  connectWiFi();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✓ WiFi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("✗ WiFi connection failed!");
    buzzLong();
    enterDeepSleep();
  }

  if (!mpu.begin()) {
    Serial.println("✗ IMU not found");
    buzzLong();
    enterDeepSleep();
  }

  Serial.println("✓ MPU6050 initialized");
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);

  Serial.println("\n🚀 System ready\n");

  // Only wait for button on first power-up
  if (!wokeFromSleep) {
    waitForButtonPress();
  } else {
    Serial.println("Resuming...\n");
  }

  delay(100);
}

// Continuous buzzer during calibration
void startCalibrationBuzzer() {
  digitalWrite(BUZZER_PIN, HIGH); // Turn buzzer ON continuously
}

void stopCalibrationBuzzer() {
  digitalWrite(BUZZER_PIN, LOW); // Turn buzzer OFF
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

  // Build JSON payload
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
  http.setTimeout(5000);
  http.setReuse(true);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Connection", "keep-alive");

  int code = http.POST((uint8_t*)payload.c_str(), payload.length());

  // Check response for sleep command
  if (code == 200) {
    String resp = http.getString();

    // Debug output
    Serial.print("POST ");
    Serial.print(code);
    Serial.print(" len=");
    Serial.print(batch_idx);
    Serial.print(" resp=");
    Serial.println(resp);

    // Check if server is calibrating
    if (resp.indexOf("calibrating") > 0) {
      Serial.println("CALIBRATING...");
      startCalibrationBuzzer();
    // Check for "Jump" command in response
    } else if (resp.indexOf("\"command\":\"JUMP\"") > 0 || resp.indexOf("\"sleep\":true") > 0) {
      http.end();
      stopCalibrationBuzzer();
      delay(100);
      buzzPattern(3, 200, 200);
      enterDeepSleep();  // Go to sleep!
      // Never reaches here
    } else {
      stopCalibrationBuzzer();
    }

  } else {  // POST failed
    Serial.print("POST failed with code: ");
    Serial.println(code);
    http.end();
    buzzLong();
    enterDeepSleep();  // Go to sleep on failure!
    // Never reaches here
  }

  http.end();
  batch_idx = 0;  // reset
}

void loop() {
  // Only collect data after button press
  if (!dataCollectionStarted) {
    delay(100);
    return;
  }

  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  Sample s;
  s.ts = millis();
  s.gx = gyro.gyro.x;  // rad/s
  s.gy = gyro.gyro.y;
  s.gz = gyro.gyro.z;
  s.ax = accel.acceleration.x;  // m/s^2
  s.ay = accel.acceleration.y;
  s.az = accel.acceleration.z;

  batch[batch_idx++] = s;
  if (batch_idx >= BATCH_N) {
    postBatch();
  }

  delay(SAMPLE_DELAY_MS);
}