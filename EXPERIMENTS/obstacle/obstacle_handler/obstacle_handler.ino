/* ESP32 Ultrasonic Obstacle + Buzzer + Blynk notify
   Hardware:
   - Ultrasonic HC-SR04 (VCC, GND, TRIG, ECHO)
   - Active buzzer (VCC -> buzzer pin HIGH to ring). If you have passive buzzer, use tone().
   - ESP32

   Replace BLYNK_AUTH_TOKEN, WIFI_SSID and WIFI_PASS.
*/

#define BLYNK_TEMPLATE_ID "TMPL6-CBvBCjX"
#define BLYNK_TEMPLATE_NAME "Warning"
#define BLYNK_AUTH_TOKEN "ZxyWLkKkMRjecqgKgqgA4rTrGQrCJItk"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

char ssid[] = "Galaxy A53 5G225D";
char pass[] = "sdci3924";

const int PIN_TRIG = 26;
const int PIN_ECHO = 25;

const int PIN_BUZZER_PASSIVE = 23;
const int PIN_BUZZER_ACTIVE  = 27;
const int PIN_LED_RED        = 22;  // Blink while has obstacle
const int PIN_LED_GREEN      = 21;  // ON when no obstacle
const int MOTOR_STOP_PIN     = 4;  // HIGH = stop motors (change to any free pin)

// ===== Distance thresholds (cm) =====
const float DIST_THRESHOLD_CM = 40.0;
const float DIST_HYSTERESIS_CM = 10.0;

// ===== Timings =====
// const unsigned long NOTIFY_REPEAT_MS = 1000UL; // every 1 seconds trigger the Blynk Push Notification
const unsigned long NOTIFY_REPEAT_MS = 5000UL; // every 5 seconds re-notify if blocked
const unsigned long SAMPLE_MS = 200;
const unsigned long BEEP_ON_MS = 300;
const unsigned long BEEP_OFF_MS = 200;
const unsigned long RED_BLINK_MS = 400;

BlynkTimer timer;
bool obstacle = false;
unsigned long lastNotifyTime = 0;
unsigned long lastSampleTime = 0;
unsigned long lastBlinkTime = 0;
bool redLedState = false;

float readUltrasonicCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long duration = pulseIn(PIN_ECHO, HIGH, 30000UL);
  if (duration == 0) return 9999.0;
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

// void beepBothBuzzers() {
//   digitalWrite(PIN_BUZZER_ACTIVE, HIGH);
//   tone(PIN_BUZZER_PASSIVE, 1000);
//   delay(BEEP_ON_MS);
//   digitalWrite(PIN_BUZZER_ACTIVE, LOW);
//   noTone(PIN_BUZZER_PASSIVE);
//   delay(BEEP_OFF_MS);
// }

void beepBothBuzzers() {
  digitalWrite(PIN_BUZZER_ACTIVE, HIGH);

  // alternating high–low pitch siren pattern for passive buzzer
  tone(PIN_BUZZER_PASSIVE, 1500); // high tone
  delay(150);
  tone(PIN_BUZZER_PASSIVE, 800);  // low tone
  delay(150);
  noTone(PIN_BUZZER_PASSIVE);

  digitalWrite(PIN_BUZZER_ACTIVE, LOW);
  delay(200);
}

// void beepBothBuzzers() {
//   static unsigned long lastBeepTime = 0;
//   static bool isOn = false;
//   unsigned long now = millis();

//   if (isOn && now - lastBeepTime >= BEEP_ON_MS) {
//     // Turn OFF both
//     digitalWrite(PIN_BUZZER_ACTIVE, LOW);
//     noTone(PIN_BUZZER_PASSIVE);
//     isOn = false;
//     lastBeepTime = now;
//   } 
//   else if (!isOn && now - lastBeepTime >= BEEP_OFF_MS) {
//     // Turn ON both
//     digitalWrite(PIN_BUZZER_ACTIVE, HIGH);
//     tone(PIN_BUZZER_PASSIVE, 2000); // try 2 kHz for clear audible beep
//     isOn = true;
//     lastBeepTime = now;
//   }
// }

void blinkRedLED() {
  unsigned long now = millis();
  if (now - lastBlinkTime >= RED_BLINK_MS) {
    redLedState = !redLedState;
    digitalWrite(PIN_LED_RED, redLedState);
    lastBlinkTime = now;
  }
}

void checkObstacle() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_MS) return;
  lastSampleTime = now;

  float dist = readUltrasonicCm();
  Serial.printf("[ULTRA] Distance: %.2f cm\n", dist);

  if (!obstacle && dist <= DIST_THRESHOLD_CM) {
    obstacle = true;
    Serial.println("Obstacle detected!");
    // sendNotifyOnce("🚨 Your trolley has stopped! An obstacle is blocking its path. Please turn back and clear it.");
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(MOTOR_STOP_PIN, HIGH);
  }
  else if (obstacle && dist > (DIST_THRESHOLD_CM + DIST_HYSTERESIS_CM)) {
    obstacle = false;
    Serial.println("Path cleared!");
    digitalWrite(PIN_LED_RED, LOW);
    digitalWrite(PIN_LED_GREEN, HIGH);
    // analogWrite(PIN_LED_GREEN, 255); 
    digitalWrite(MOTOR_STOP_PIN, LOW);
  }

  if (obstacle) {
    blinkRedLED();
    beepBothBuzzers();
  } else {
    digitalWrite(PIN_LED_RED, LOW);
    digitalWrite(PIN_LED_GREEN, HIGH);
    // analogWrite(PIN_LED_GREEN, 255); 
  }
}

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("ESP32 Ultrasonic + Dual Buzzer + Dual LED + Motor Stop");

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_BUZZER_PASSIVE, OUTPUT);
  pinMode(PIN_BUZZER_ACTIVE, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(MOTOR_STOP_PIN, OUTPUT);

  digitalWrite(PIN_BUZZER_ACTIVE, LOW);
  noTone(PIN_BUZZER_PASSIVE);
  digitalWrite(PIN_LED_RED, LOW);
  digitalWrite(PIN_LED_GREEN, HIGH);
  digitalWrite(MOTOR_STOP_PIN, LOW);

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  timer.setInterval(100L, checkObstacle);
}

void loop() {
  Blynk.run();
  timer.run();
}