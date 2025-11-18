#include <WiFi.h>
#include "ESP32MQTTClient.h"

// Replace with your phone's hotspot credentials
const char* ssid = "InsertHere";
const char* password = "InsertHere";

// Replace with your Windows IP (from `ipconfig`)
const char* mqtt_server = "mqtt://InsertHere:1883";  // Use full URI format

ESP32MQTTClient mqttClient;

void setup_wifi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void setup_mqtt() {
  mqttClient.setURI(mqtt_server);
  mqttClient.enableDebuggingMessages();
  mqttClient.setKeepAlive(30);
  mqttClient.enableLastWillMessage("test/lwt", "ESP32 disconnected");

  mqttClient.setOnMessageCallback([](const std::string &topic, const std::string &payload) {
    Serial.print("Message received on ");
    Serial.print(topic.c_str());
    Serial.print(": ");
    Serial.println(payload.c_str());
  });
}

bool connect_mqtt() {
  Serial.println("Connecting to MQTT...");
  mqttClient.loopStart();  // Start background loop

  int attempts = 0;
  const int maxAttempts = 20;

  while (!mqttClient.isConnected() && attempts < maxAttempts) {
    Serial.print(".");
    delay(500);
    attempts++;
  }

  Serial.println();

  if (mqttClient.isConnected()) {
    Serial.println("MQTT connected!");

    // Subscribe with full signature
    mqttClient.subscribe("test/topic",
      [](const std::string &payload) {
        Serial.print("Received message: ");
        Serial.println(payload.c_str());
      },
      0  // QoS level
    );

    // Publish a test message
    mqttClient.publish("test/topic", "Hello from ESP32!", 0, false);
    return true;
  }

  Serial.println("MQTT connection failed.");
  return false;
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  setup_mqtt();
  connect_mqtt();
}

void loop() {
  // Nothing needed here — mqttClient.loopStart() handles background tasks
}

void onMqttConnect(esp_mqtt_client_handle_t client) {
  Serial.println("MQTT connection established (onMqttConnect)");
}

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