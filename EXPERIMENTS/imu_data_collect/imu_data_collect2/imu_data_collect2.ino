#include "Arduino.h"
#include <WiFi.h>
#include "ESP32MQTTClient.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "esp_idf_version.h"

const int buttonPin = 14;
const int buzzerPin = 26;
const int ledPin = 13;

// WIFI CREDENTIALS
const char *ssid = "---";
const char *pass = "---";

// MQTT Configuration
char *server = "mqtt://---:1883";
char *publishTopicIMU = "eve/imu";
char *publishTopicButton = "eve/button";
char *subscribeTopicControl = "eve/control";

// Debounce settings
const unsigned long debounceDelay = 80;
int stableState = HIGH;
int lastReading = HIGH;
unsigned long lastDebounceTime = 0;

// State variables
bool isConnected = false;
bool isRecording = false;
unsigned long lastConnectionCheck = 0;
const unsigned long connectionCheckInterval = 5000;

ESP32MQTTClient mqttClient;
Adafruit_MPU6050 mpu;

// Required MQTT callback functions
void onMqttConnect(esp_mqtt_client_handle_t client) {
  Serial.println("MQTT Connected callback triggered");
  isConnected = true;
}

void onMqttDisconnect(esp_mqtt_client_handle_t client) {
  Serial.println("MQTT Disconnected callback triggered");
  isConnected = false;
  isRecording = false;
  digitalWrite(ledPin, LOW);
}

void onMqttSubscribe(esp_mqtt_client_handle_t client, int msgId) {
  Serial.println("MQTT Subscribe acknowledged");
}

void onMqttUnsubscribe(esp_mqtt_client_handle_t client, int msgId) {
  Serial.println("MQTT Unsubscribe acknowledged");
}

void onMqttPublish(esp_mqtt_client_handle_t client, int msgId) {
  // Published message acknowledged
}

void onMqttData(esp_mqtt_client_handle_t client, const std::string &topic, const std::string &payload) {
  Serial.print("MQTT Message - ");
  Serial.print(topic.c_str());
  Serial.print(": ");
  Serial.println(payload.c_str());
  
  if (topic == "eve/control") {
    if (payload == "stop_recording") {
      Serial.println("Stop recording command received");
      isRecording = false;
      digitalWrite(ledPin, LOW);
      tone(buzzerPin, 300, 80);
    } else if (payload == "ready") {
      Serial.println("Python ready to receive data");
    }
  }
}

void onMqttError(esp_mqtt_client_handle_t client, int error) {
  Serial.print("MQTT Error: ");
  Serial.println(error);
}

// MQTT Event Handler
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
esp_err_t handleMQTT(esp_mqtt_event_handle_t event) {
  mqttClient.onEventCallback(event);
  return ESP_OK;
}
#else
void handleMQTT(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
  auto *event = static_cast<esp_mqtt_event_handle_t>(event_data);
  mqttClient.onEventCallback(event);
}
#endif

void buzzPattern(int count, int duration = 200, int pause = 300) {
  for (int i = 0; i < count; i++) {
    tone(buzzerPin, 2000, duration);
    delay(duration + pause);
  }
}

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
    onMqttData(nullptr, topic, payload);
  });
}

bool connectWiFi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  WiFi.setHostname("eve-imu-device");
  
  int attempts = 0;
  const int maxAttempts = 20;
  
  while (WiFi.status() != WL_CONNECTED && attempts < maxAttempts) {
    Serial.print(".");
    delay(500);
    attempts++;
  }
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected! IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }
  
  Serial.println("WiFi connection failed!");
  return false;
}

bool connectToMQTT() {
  Serial.println("Starting MQTT connection...");
  
  mqttClient.loopStart();
  
  int attempts = 0;
  const int maxAttempts = 20;
  
  while (!mqttClient.isConnected() && attempts < maxAttempts) {
    Serial.print(".");
    delay(500);
    attempts++;
  }
  Serial.println();
  
  if (mqttClient.isConnected()) {
    Serial.println("MQTT Connected!");
    delay(500);
    
    mqttClient.subscribe(subscribeTopicControl, [](const std::string &payload) {
      Serial.println("Control message via subscription: " + String(payload.c_str()));
    });
    
    Serial.println("Subscribed to control topic");
    return true;
  }
  
  Serial.println("MQTT connection failed!");
  return false;
}

void sendIMUData() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // JSON string with timestamp and IMU data
  String imuData = String("{") +
    "\"timestamp\":" + String(millis()) + "," +
    "\"gx\":" + String(g.gyro.x, 4) + "," +
    "\"gy\":" + String(g.gyro.y, 4) + "," +
    "\"gz\":" + String(g.gyro.z, 4) + "," +
    "\"ax\":" + String(a.acceleration.x, 4) + "," +
    "\"ay\":" + String(a.acceleration.y, 4) + "," +
    "\"az\":" + String(a.acceleration.z, 4) +
    "}";
  
  mqttClient.publish(publishTopicIMU, imuData.c_str(), 0, false);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  
  // Initialize MPU6050
  Wire.begin(21, 22);
  Serial.println("Initializing MPU6050...");
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("ERROR: MPU6050 connection failed!");
    buzzPattern(5, 200, 200);
    while(1) delay(1000);
  }
  Serial.println("MPU6050 connected successfully");
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  // Setup MQTT
  setupMQTT();
  
  // Connect to WiFi and MQTT
  if (connectWiFi() && connectToMQTT()) {
    isConnected = true;
    buzzLong();
    Serial.println("Device ready. Press button to start recording...");
  } else {
    buzzPattern(3);
    Serial.println("Connection failed. Please reset device.");
  }
  
  delay(500);
}

void loop() {
  // Read button with debouncing
  int reading = digitalRead(buttonPin);
  
  if (reading != lastReading) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != stableState) {
      stableState = reading;
      
      // Button pressed (LOW) and connected but not recording
      if (stableState == LOW && isConnected && !isRecording) {
        Serial.println("Button pressed - Starting recording...");
        
        // Send start command to Python
        mqttClient.publish(publishTopicButton, "start_recording", 0, false);
        
        digitalWrite(ledPin, HIGH);
        tone(buzzerPin, 1000, 120);
        
        isRecording = true;
        Serial.println("Recording started - Streaming data at 50Hz");
      }
    }
  }
  
  lastReading = reading;
  
  // If recording, send IMU data at 50Hz
  if (isRecording && isConnected) {
    sendIMUData();
    
    // Check MQTT connection periodically
    if (millis() - lastConnectionCheck > connectionCheckInterval) {
      if (!mqttClient.isConnected()) {
        Serial.println("MQTT connection lost!");
        buzzPattern(3);
        isConnected = false;
        isRecording = false;
        digitalWrite(ledPin, LOW);
        WiFi.disconnect(true);
        return;
      }
      lastConnectionCheck = millis();
    }
    
    delay(20); // 50Hz = 20ms delay
  } else {
    delay(100); // Check button less frequently when not recording
  }
}