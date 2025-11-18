#include <esp_sleep.h>

#define BUTTON_PIN GPIO_NUM_33  
#define LED_PIN 2               
int lastButtonPressed = 0;
int currentTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  
  esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();
  
  if (wakeup_reason == ESP_SLEEP_WAKEUP_EXT0) {
    Serial.println("Woken up by button press");
  } else {
    Serial.println("Normal startup or reset");
  }
  
  lastButtonPressed = millis(); // Initialize the debounce timer
  esp_sleep_enable_ext0_wakeup(BUTTON_PIN, 0);
}

void loop() {  
  currentTime = millis();
  if (currentTime - lastButtonPressed >= 150 && digitalRead(BUTTON_PIN) == LOW) {
    Serial.println("Button pressed - entering deep sleep");
    delay(100);  
      
    esp_deep_sleep_start(); 
  }
}