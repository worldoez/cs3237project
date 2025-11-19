#define BLYNK_TEMPLATE_ID "TMPL6-CBvBCjX"
#define BLYNK_TEMPLATE_NAME "Warning"
#define BLYNK_AUTH_TOKEN "ZxyWLkKkMRjecqgKgqgA4rTrGQrCJItk"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

char ssid[] = "";
char pass[] = "";

const int PIN_TRIG = 26;
const int PIN_ECHO = 25;

const int PIN_BUZZER_PASSIVE = 14;
const int PIN_BUZZER_ACTIVE  = 27;
const int MOTOR_STOP_PIN     = 4;  // HIGH = stop motors

// Distance thresholds (cm)
const float DIST_THRESHOLD_CM   = 30.0;
const float DIST_HYSTERESIS_CM  = 10.0;

// Timings
const unsigned long NOTIFY_REPEAT_MS = 5000UL;
const unsigned long SAMPLE_MS        = 200;
const unsigned long BEEP_ON_MS       = 300;
const unsigned long BEEP_OFF_MS      = 200;

BlynkTimer timer;
bool obstacle = false;
unsigned long lastNotifyTime = 0;
unsigned long lastSampleTime = 0;

float readUltrasonicCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long duration = pulseIn(PIN_ECHO, HIGH, 30000UL);
  if (duration == 0) return 9999.0;         // timeout -> treat as far
  return (duration * 0.0343) / 2.0;
}

void sendNotifyOnce(const String &msg) {
  unsigned long now = millis();
  if (now - lastNotifyTime >= NOTIFY_REPEAT_MS) {
    Blynk.logEvent("obstacle_alert", msg);
    lastNotifyTime = now;
    Serial.println("[BLYNK] Notified: " + msg);
  }
}

void beepBothBuzzers() {
  digitalWrite(PIN_BUZZER_ACTIVE, HIGH);
  tone(PIN_BUZZER_PASSIVE, 1500);
  delay(150);
  tone(PIN_BUZZER_PASSIVE, 800);
  delay(150);
  noTone(PIN_BUZZER_PASSIVE);
  digitalWrite(PIN_BUZZER_ACTIVE, LOW);
  delay(200);
}

void checkObstacle() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_MS) return;
  lastSampleTime = now;

  float dist = readUltrasonicCm();
  Serial.printf("[ULTRA] Distance: %.2f cm\n", dist);

  // Enter obstacle state
  if (!obstacle && dist <= DIST_THRESHOLD_CM) {
    obstacle = true;
    Serial.println("Obstacle detected!");
    sendNotifyOnce("Your trolley has stopped! An obstacle is blocking its path. Please clear it.");
    digitalWrite(MOTOR_STOP_PIN, HIGH);
  }
  // Exit obstacle state (hysteresis)
  else if (obstacle && dist > (DIST_THRESHOLD_CM + DIST_HYSTERESIS_CM)) {
    obstacle = false;
    Serial.println("Path cleared!");
    digitalWrite(MOTOR_STOP_PIN, LOW);
  }

  if (obstacle) {
    beepBothBuzzers();
  }
}

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("ESP32 Ultrasonic + Dual Buzzer + Motor Stop (LEDs removed)");

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_BUZZER_PASSIVE, OUTPUT);
  pinMode(PIN_BUZZER_ACTIVE, OUTPUT);
  pinMode(MOTOR_STOP_PIN, OUTPUT);

  digitalWrite(PIN_BUZZER_ACTIVE, LOW);
  noTone(PIN_BUZZER_PASSIVE);
  digitalWrite(MOTOR_STOP_PIN, LOW);

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  timer.setInterval(100L, checkObstacle);
}

void loop() {
  Blynk.run();
  timer.run();
}