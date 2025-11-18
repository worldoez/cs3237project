#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "base64.h"
#include "board_config.h"

// WiFi credentials
const char* ap_ssid = "UGLL-E-CAM-WIFI";
const char* ap_password = "88888888";

// MQTT Configuration
const char* mqtt_server = "192.168.4.2";  // Change to your MQTT broker IP
const int mqtt_port = 1883;
const char* mqtt_client_id = "esp32_cam01";
const char* device_id = "cam01";

// MQTT Topics
const char* topic_cam = "/cam";
const char* topic_cam_out = "/cam_out";

// Global objects
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

// Timing
unsigned long lastPublish = 0;
const long publishInterval = 100;  // Publish every 100ms (10 Hz)

void setupLedFlash();
void reconnect_mqtt();
void callback(char* topic, byte* payload, unsigned int length);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 16000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_240X240;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 20;
  config.fb_count = 2;

  // Camera initialization
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  
  // Adjust sensor settings
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  // WiFi AP setup
  WiFi.softAP(ap_ssid, ap_password);
  WiFi.setSleep(false);

  Serial.println("");
  Serial.print("WiFi AP started. Connect to: ");
  Serial.println(ap_ssid);
  Serial.print("Password: ");
  Serial.println(ap_password);
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());

  // MQTT setup
  mqtt_client.setServer(mqtt_server, mqtt_port);
  mqtt_client.setCallback(callback);
  mqtt_client.setBufferSize(65536);  // Increase buffer for image data

  Serial.println("Camera Ready!");
}

void reconnect_mqtt() {
  // Loop until we're reconnected
  while (!mqtt_client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    if (mqtt_client.connect(mqtt_client_id)) {
      Serial.println("connected");
      // Subscribe to output topic for acknowledgments/commands
      mqtt_client.subscribe(topic_cam_out);
      Serial.print("Subscribed to: ");
      Serial.println(topic_cam_out);
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqtt_client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  // Handle incoming messages on /cam_out
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  
  // Parse JSON
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, payload, length);
  
  if (error) {
    Serial.print("JSON parse failed: ");
    Serial.println(error.c_str());
    return;
  }
  
  // Extract information
  const char* dev_id = doc["device_id"];
  float distance = doc["distance"];
  bool is_present = doc["is_apriltag_present"];
  
  Serial.printf("Device: %s, Distance: %.2f cm, Tag present: %s\n", 
                dev_id, distance, is_present ? "Yes" : "No");
}

void publishCameraFrame() {
  // Capture frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // Encode to base64
  String encoded = base64::encode(fb->buf, fb->len);
  
  // Create JSON document
  StaticJsonDocument<65536> doc;
  doc["device_id"] = device_id;
  doc["image"] = encoded;
  doc["timestamp"] = millis();
  doc["width"] = fb->width;
  doc["height"] = fb->height;
  doc["format"] = "grayscale";

  // Serialize JSON
  String output;
  serializeJson(doc, output);

  // Publish to MQTT
  bool success = mqtt_client.publish(topic_cam, output.c_str(), false);
  
  if (success) {
    Serial.printf("Published frame (%d bytes, encoded: %d bytes)\n", 
                  fb->len, encoded.length());
  } else {
    Serial.println("Failed to publish frame");
  }

  // Return frame buffer
  esp_camera_fb_return(fb);
}

void loop() {
  // Maintain MQTT connection
  if (!mqtt_client.connected()) {
    reconnect_mqtt();
  }
  mqtt_client.loop();

  // Publish frames at specified interval
  unsigned long currentMillis = millis();
  if (currentMillis - lastPublish >= publishInterval) {
    lastPublish = currentMillis;
    publishCameraFrame();
  }
}

void setupLedFlash() {
  // LED flash setup (if needed)
  #if defined(LED_GPIO_NUM)
    pinMode(LED_GPIO_NUM, OUTPUT);
  #endif
}