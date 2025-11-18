#include "Arduino.h"
#include <WiFi.h>
#include "ESP32MQTTClient.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "esp_idf_version.h"

const int buttonPin = 14;
const int buzzerPin = 26;

// WIFI CREDENTIALS -----------------------
const char *ssid = "-";
const char *pass = "-";
// ----------------------------------------

char *server = "mqtt://192.168.78.152:1883";
char *publishTopicIMU = "eve/imu";
char *publishTopicButton = "eve/button";
char *subscribeTopicSleep = "eve/sleep";

// Deep sleep configuration
#define uS_TO_S_FACTOR 1000000ULL
RTC_DATA_ATTR int bootCount = 0;

// State variables
bool isActive = false;
bool shouldSleep = false;
unsigned long lastConnectionCheck = 0;
const unsigned long connectionCheckInterval = 5000; // Check every 5 seconds

// Debounce settings
const unsigned long debounceDelay = 80;
int stableState = HIGH;
int lastReading = HIGH;
unsigned long lastDebounceTime = 0;

ESP32MQTTClient mqttClient;
Adafruit_MPU6050 mpu;

// BUZZER: 3 short buzzes
void buzzPattern(int count, int duration = 200, int pause = 300) {
  for (int i = 0; i < count; i++) {
    tone(buzzerPin, 2000, duration);
    delay(duration + pause);
  }
}

// BUZZER: 1 long buzz
void buzzLong() {
  tone(buzzerPin, 1000, 800);
  delay(800);
}

void setupMQTT() {
  mqttClient.enableDebuggingMessages();
  mqttClient.setURI(server);
  mqttClient.enableLastWillMessage("eve/lwt", "Device offline");
  mqttClient.setKeepAlive(30);
  
  mqttClient.setOnMessageCallback([](const std::string &topic, const std::string &payload) {
    log_i("Message received - %s: %s", topic.c_str(), payload.c_str());
    
    if (topic == subscribeTopicSleep) {
      Serial.println("Sleep command received!");
      shouldSleep = true;
    }
  });
}

bool connectToMQTT() {
  Serial.println("Connecting to MQTT...");
  
  int attempts = 0;
  // ----------------------------------
  const int maxAttempts = 10;
  // ----------------------------------
  
  while (!mqttClient.isConnected() && attempts < maxAttempts) {
    delay(500);
    attempts++;
    Serial.print(".");
  }
  Serial.println();
  
  if (mqttClient.isConnected()) {
    Serial.println("MQTT Connected!");
    
    // Subscribe to sleep topic
    mqttClient.subscribe(subscribeTopicSleep, [](const std::string &payload) {
      Serial.println("Sleep command received via subscription!");
      shouldSleep = true;
    });
    
    return true;
  }
  
  return false;
}

bool connectWiFi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  WiFi.setHostname("eve-imu-device");
  
  int attempts = 0;
  // ----------------------------
  const int maxAttempts = 20;
  // ----------------------------
  
  while (WiFi.status() != WL_CONNECTED && attempts < maxAttempts) {
    Serial.print(".");
    delay(500);
    attempts++;
  }
  Serial.println();
  
  return WiFi.status() == WL_CONNECTED;
}

void sendIMUData() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // JSON-like string with all IMU data
  String imuData = String("{") +
    "\"gx\":" + String(g.gyro.x, 4) + "," +
    "\"gy\":" + String(g.gyro.y, 4) + "," +
    "\"gz\":" + String(g.gyro.z, 4) + "," +
    "\"ax\":" + String(a.acceleration.x, 4) + "," +
    "\"ay\":" + String(a.acceleration.y, 4) + "," +
    "\"az\":" + String(a.acceleration.z, 4) +
    "}";
  
  mqttClient.publish(publishTopicIMU, imuData.c_str(), 0, false);
}

void enterDeepSleep() {
  Serial.println("Disconnecting MQTT...");
  mqttClient.loopStop();
  
  Serial.println("Disconnecting WiFi...");
  WiFi.disconnect(true);
  
  Serial.println("Configuring wake on button press...");
  esp_sleep_enable_ext0_wakeup((gpio_num_t)buttonPin, 0); // Wake on LOW (button press)
  
  Serial.println("Entering deep sleep...");
  Serial.flush();
  esp_deep_sleep_start();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT);
  
  bootCount++;
  Serial.println("Boot count: " + String(bootCount));
  
  // Initialize MPU6050
  Wire.begin(21, 22);
  Serial.println("Initializing MPU6050...");
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR: MPU6050 connection failed!");
    buzzPattern(5, 200, 200);
    esp_deep_sleep_start();
  }
  Serial.println("MPU6050 connected successfully");
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  // Setup MQTT
  setupMQTT();
  
  // Wait for button to be released before continuing (debounce on wake)
  delay(500);
  
  Serial.println("Device ready. Press button to start...");
}

// Maybe this architecture could be improved? ******************************************
void loop() {
  // Read button with debouncing
  int reading = digitalRead(buttonPin);
  
  if (reading != lastReading) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading;
      
      // Button pressed (LOW) and not active - try to connect
      if (stableState == LOW && !isActive) {
        Serial.println("Button pressed - attempting connection...");
        
        // Try to connect to WiFi
        if (!connectWiFi()) {
          Serial.println("WiFi connection failed!");
          buzzPattern(3);
          lastReading = reading;
          return;
        }
        
        Serial.println("WiFi connected!");
        
        // Start MQTT loop
        mqttClient.loopStart();
        
        // Try to connect to MQTT
        if (!connectToMQTT()) {
          Serial.println("MQTT connection failed!");
          buzzPattern(3);
          WiFi.disconnect(true);
          lastReading = reading;
          return;
        }
        
        // Success!
        Serial.println("Connected successfully!");
        buzzLong();
        
        // Send start command
        mqttClient.publish(publishTopicButton, "start", 0, false);
        Serial.println("Sent 'start' command");
        
        isActive = true;
        lastConnectionCheck = millis();
      }
    }
  }
  
  lastReading = reading;
  
  // If active, send IMU data and check connection
  if (isActive) {
    // Send IMU data continuously
    sendIMUData();
    
    // Check MQTT connection periodically
    if (millis() - lastConnectionCheck > connectionCheckInterval) {
      if (!mqttClient.isConnected()) {
        Serial.println("MQTT connection lost!");
        buzzPattern(3);
        isActive = false;
        WiFi.disconnect(true);
        return;
      }
      lastConnectionCheck = millis();
    }
    
    // Check if sleep command received
    if (shouldSleep) {
      Serial.println("Sleep command detected, entering sleep mode...");
      enterDeepSleep();
    }
    
    delay(50); // Send IMU data at ~20Hz

  } else {
    delay(100); // Check button less frequently when inactive
    
  }
}