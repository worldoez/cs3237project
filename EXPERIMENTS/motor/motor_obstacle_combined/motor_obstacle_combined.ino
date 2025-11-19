// A -> Back Left
// B -> Back Right
// C -> Front Right
// D -> Front Left

#define BLYNK_TEMPLATE_ID "TMPL6-CBvBCjX"
#define BLYNK_TEMPLATE_NAME "Warning"
#define BLYNK_AUTH_TOKEN "ZxyWLkKkMRjecqgKgqgA4rTrGQrCJItk"

#include <WiFi.h>
#include <HTTPClient.h>
#include <esp_sleep.h>
#include <BlynkSimpleEsp32.h>

#define BUTTON_PIN GPIO_NUM_33  
RTC_DATA_ATTR int lastButtonPressed = 0;
RTC_DATA_ATTR int currentTime = 0;
volatile bool buttonPressedFlag = false;

//motor A connected between A01 and A02
//motor B connected between B01 and B02
int STBY1 = 26; //standby
int STBY2 = 27; //standby
//Motor A
int PWMA = 5;  //Speed control
int AIN1 = 17; //Direction
int AIN2 = 16; //Direction
//Motor B
int PWMB = 14; 
int BIN1 = 13; 
int BIN2 = 12; 
//Motor C
int PWMC = 21;
int CIN1 = 18;
int CIN2 = 19;
//Motor D
int PWMD = 15;
int DIN1 = 0;
int DIN2 = 2;

//RED LED
int RED_LED = 25;

//WIFI LED
int WIFI_LED = 32;

// Stop-signal input from obstacle ESP32
bool obstacleActive = false;
const int PIN_TRIG = 4;
const int PIN_ECHO = 35;

const int PIN_BUZZER_PASSIVE = 23;
const int PIN_BUZZER_ACTIVE  = 22;

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
// unsigned long lastBlinkTime = 0;
bool redLedState = false;


String currentCommand = "0";
int motorSpeed = 255;

#define STOP "0"
#define FORWARD "1"
#define BACKWARD "2"
#define TURN_LEFT "3"
#define TURN_RIGHT "4"
#define SIDE_LEFT "5"
#define SIDE_RIGHT "6"
#define FULL_TURN "7"
#define JUMP "8"
#define TOO_FAR "9"
#define TOO_NEAR "10"
#define OBSTACLE "11"

#define TIME 1000
#define ROTATE_TIME 1350

void moveForward(int moveTime);
void moveBackward(int moveTime);
void moveSideLeft(int moveTime);
void moveSideRight(int moveTime);
void moveTurnLeft(int moveTime);
void moveTurnRight(int moveTime);
void moveRotate(int moveTime);
void testMapping();
void offAllMotor();
float readUltrasonicCm();
void sendNotifyOnce(const String &msg);
void checkObstacle();
void beepBothBuzzers();

// http
// const char* ssid = "Galaxy A53 5G225D";
// const char* password = "sdci3924";
// const char* serverName = "http://10.81.21.177:5000/control";
// const char* ssid = "UGLEE-CAM-WIFI";
const char* ssid = "aaaaaaaa";
const char* password = "88888888";
// const char* serverName = "http://10.235.243.246:5000/";
const char* serverName = "http://192.168.4.9:5002/fetchData";

void IRAM_ATTR isr() {
    buttonPressedFlag = true;
}

void setup() {
    Serial.begin(115200);

    pinMode(STBY1, OUTPUT);
    pinMode(STBY2, OUTPUT);
    pinMode(PWMA, OUTPUT);
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(PWMB, OUTPUT);
    pinMode(BIN1, OUTPUT);
    pinMode(BIN2, OUTPUT);
    pinMode(PWMC, OUTPUT);
    pinMode(CIN1, OUTPUT);
    pinMode(CIN2, OUTPUT);
    pinMode(PWMD, OUTPUT);
    pinMode(DIN1, OUTPUT);
    pinMode(DIN2, OUTPUT);
    pinMode(RED_LED, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(WIFI_LED, OUTPUT);
    pinMode(PIN_TRIG, OUTPUT);
    pinMode(PIN_ECHO, INPUT);
    pinMode(PIN_BUZZER_PASSIVE, OUTPUT);
    pinMode(PIN_BUZZER_ACTIVE, OUTPUT);

    digitalWrite(WIFI_LED, LOW);
    digitalWrite(PIN_BUZZER_ACTIVE, LOW);
    noTone(PIN_BUZZER_PASSIVE);

    esp_sleep_wakeup_cause_t wakeup_reason = esp_sleep_get_wakeup_cause();

    if (wakeup_reason == ESP_SLEEP_WAKEUP_EXT0) {
        Serial.println("Woken up by button press");
    
        // wait for button release
        while(digitalRead(BUTTON_PIN) == LOW) {
            delay(10);
        }
        delay(100); // Extra debounce
    } else {
        Serial.println("Normal startup or reset");
    }

    esp_sleep_enable_ext0_wakeup(BUTTON_PIN, 0);

    // Attach interrupt for during operation
    attachInterrupt(BUTTON_PIN, isr, FALLING);
    lastButtonPressed = millis();

    // WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
  
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    digitalWrite(WIFI_LED, HIGH);

    //set motor A and motor B speed, 0-255 255 being the fastest
    analogWrite(PWMA, motorSpeed);
    analogWrite(PWMB, motorSpeed);
    analogWrite(PWMC, motorSpeed);
    analogWrite(PWMD, motorSpeed);

    // Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
    // timer.setInterval(100L, checkObstacle);
}

void loop() {
    Blynk.run();
    timer.run();

    currentTime = millis();

    // static bool lastObstacleState = false;  // tracks previous signal state

    checkObstacle();

    // // Obstacle newly detected (HIGH after LOW)
    // if (obstacleActive && !lastObstacleState) {
    //     Serial.println("Obstacle detected — slowing down...");
    //     gradualStop();
    // }

    // // Obstacle cleared (LOW after HIGH)
    // if (!obstacleActive && lastObstacleState) {
    //     Serial.println("Path cleared — resuming smoothly...");
    //     gradualStart();
    // }
    // lastObstacleState = obstacleActive;  // update for next cycle

    // // If obstacle is still present, do nothing
    // if (obstacleActive) return;

    // HTTP
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(WIFI_LED, HIGH);
        HTTPClient http;

        http.begin(serverName);
        int httpResponseCode = http.GET();
        
        if (httpResponseCode > 0) {
            // Serial.print("HTTP Response code: ");
            // Serial.println(httpResponseCode);
            
            // Get the response payload
            String command = http.getString();
            if (command != currentCommand) {
                currentCommand = command;
                Serial.print("New command: ");
                Serial.println(currentCommand);
            }
        }
        else {
            Serial.print("Error code: ");
            currentCommand = "0";
            Serial.println(httpResponseCode);
        }

        if (obstacleActive) currentCommand = OBSTACLE;

        http.end();
    } else {
        Serial.println("WiFi Disconnected");
        digitalWrite(WIFI_LED, LOW);
    }
    // testMapping();

    // Check for deep sleep request
    if (buttonPressedFlag) {
        buttonPressedFlag = false;
        
        if (currentTime - lastButtonPressed >= 150) {
            // Wait for button release before sleeping
            while(digitalRead(BUTTON_PIN) == LOW) {
                delay(10);
            }
            
            Serial.println("Button pressed - entering deep sleep");
            Serial.flush();
            delay(100);
            
            // Detach interrupt before sleep
            detachInterrupt(BUTTON_PIN);
            
            esp_deep_sleep_start();
        }
        lastButtonPressed = currentTime;
    }

    executeCommand(currentCommand);
    delay(50);
}

// Gradually reduce motor speed before stopping
void gradualStop() {
    analogWrite(PWMA, motorSpeed * 0.7);
    analogWrite(PWMB, motorSpeed * 0.7);
    analogWrite(PWMC, motorSpeed * 0.7);
    analogWrite(PWMD, motorSpeed * 0.7);
    delay(200);
    analogWrite(PWMA, motorSpeed * 0.5);
    analogWrite(PWMB, motorSpeed * 0.5);
    analogWrite(PWMC, motorSpeed * 0.5);
    analogWrite(PWMD, motorSpeed * 0.5);
    delay(200);
    offAllMotor();
    Serial.println("Motors stopped completely.");
}

// Gradually increase motor speed from stop to full
void gradualStart() {
    // ramp up from standstill to full power
    analogWrite(PWMA, motorSpeed * 0.5);
    analogWrite(PWMB, motorSpeed * 0.5);
    analogWrite(PWMC, motorSpeed * 0.5);
    analogWrite(PWMD, motorSpeed * 0.5);
    delay(200);
    analogWrite(PWMA, motorSpeed * 0.7);
    analogWrite(PWMB, motorSpeed * 0.7);
    analogWrite(PWMC, motorSpeed * 0.7);
    analogWrite(PWMD, motorSpeed * 0.7);
    delay(200);
    analogWrite(PWMA, motorSpeed);
    analogWrite(PWMB, motorSpeed);
    analogWrite(PWMC, motorSpeed);
    analogWrite(PWMD, motorSpeed);
    delay(100);
    Serial.println("Motors resumed to full speed.");
}

void executeCommand(String command) {
    if (command == JUMP) {
        offAllMotor();
        Serial.println("jump");
        digitalWrite(RED_LED, HIGH);
        delay(500);
        for(;;);
    }
    else if (command == TOO_FAR) {
        moveForward(TIME);
        beepBothBuzzers();
        Serial.println("too far");
    }
    else if (command == TOO_NEAR) {
        offAllMotor();
        beepBothBuzzers();
        Serial.println("too near");
    }
    else if (command == FORWARD) {
        moveForward(TIME);
        Serial.println("forward");
    }
    else if (command == BACKWARD) {
        moveBackward(TIME);
        Serial.println("backward");
    }
    else if (command == TURN_LEFT) {
        moveForward(TIME); 
        moveTurnLeft(ROTATE_TIME);
        Serial.println("rotate left");
        delay(1000);
    } 
    else if (command == TURN_RIGHT) {
        moveForward(TIME);
        moveTurnRight(ROTATE_TIME);
        Serial.println("rotate right");
        delay(1000);
    }
    else if (command == SIDE_LEFT) {
        moveSideLeft(TIME);
        Serial.println("side left");
    }
    else if (command == SIDE_RIGHT) {
        moveSideRight(TIME);
        Serial.println("side right");
    } 
    else if (command == FULL_TURN) {
        moveRotate(ROTATE_TIME);
        Serial.println("full rotate");
    }
    else if (command == STOP || command == "") {
        offAllMotor();
        Serial.println("stop");
        digitalWrite(RED_LED, HIGH);
        delay(500);
        digitalWrite(RED_LED, LOW);
        delay(500);      
    }
    else if (command = OBSTACLE) {
        offAllMotor();
        Serial.println("obstacle stop");
        delay(500);
    }
    else {
        Serial.println("Unknown command");
    }
}

void moveForward(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void moveBackward(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,HIGH);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,HIGH);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}


void moveSideLeft(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,HIGH);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void moveSideRight(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,HIGH);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void moveTurnLeft(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void moveTurnRight(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,HIGH);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,HIGH);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void moveRotate(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    delay(4*moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

void offAllMotor() {
    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,LOW);
}

void testMapping() {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);
    
    offAllMotor();
    delay(500);
    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);
    delay(1000);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

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
  }
  else if (obstacle && dist > (DIST_THRESHOLD_CM + DIST_HYSTERESIS_CM)) {
    obstacle = false;
    Serial.println("Path cleared!");
  }

  if (obstacle) {
    beepBothBuzzers();
    obstacleActive = true;

  } else {
    obstacleActive = false;
  }
}
