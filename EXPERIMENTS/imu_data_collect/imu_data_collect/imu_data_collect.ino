#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Pin definitions
const int buttonPin = 14;
const int buzzerPin = 26;
const int ledPin = 13;

// Debounce settings
const unsigned long debounceDelay = 80;
int stableState = HIGH;
int lastReading = HIGH;
unsigned long lastDebounceTime = 0;

// Recording state
bool buzzerOn = false;
bool isRecording = false;
int actionId = 0;
unsigned long recordStartTime = 0;

// Gyroscope
Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  Wire.begin(21, 22);  // SDA, SCL

  Serial.println("INIT:Initializing MPU6050...");
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR:MPU6050 connection failed!");
    while (1) {
      tone(buzzerPin, 2000, 200);
      delay(500);
    }
  }
  Serial.println("INIT:MPU6050 connected successfully");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("HEADER:action_id,timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z");
  Serial.println("READY:Press and hold button to record");
}

void loop() {
  int reading = digitalRead(buttonPin);

  if (reading != lastReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading;

      if (stableState == LOW && !buzzerOn) {
        Serial.println("STATUS:Recording started");
        digitalWrite(ledPin, HIGH);
        tone(buzzerPin, 1000, 120);
        buzzerOn = true;
        isRecording = true;
        actionId++;
        recordStartTime = millis();

      } else if (stableState == HIGH && buzzerOn) {
        Serial.print("STATUS:Recording stopped - Action ID ");
        Serial.println(actionId);
        digitalWrite(ledPin, LOW);
        tone(buzzerPin, 300, 80);
        buzzerOn = false;
        isRecording = false;
      }
    }
  }

  if (isRecording) {
    recordGyroData();
    delay(10);
  }

  lastReading = reading;
}

void recordGyroData() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  float gyroX = g.gyro.x;
  float gyroY = g.gyro.y;
  float gyroZ = g.gyro.z;

  float accelX = a.acceleration.x;
  float accelY = a.acceleration.y;
  float accelZ = a.acceleration.z;

  unsigned long timestamp = millis() - recordStartTime;

  Serial.print("DATA:");
  Serial.print(actionId);
  Serial.print(",");
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(gyroX, 4);
  Serial.print(",");
  Serial.print(gyroY, 4);
  Serial.print(",");
  Serial.print(gyroZ, 4);
  Serial.print(",");
  Serial.print(accelX, 4);
  Serial.print(",");
  Serial.print(accelY, 4);
  Serial.print(",");
  Serial.println(accelZ, 4);
}
