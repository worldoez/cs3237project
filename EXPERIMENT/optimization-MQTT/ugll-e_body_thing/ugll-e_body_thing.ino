// A -> Back Left
// B -> Back Right
// C -> Front Right
// D -> Front Left

#include <WiFi.h>
#include <PubSubClient.h>
#include <esp_sleep.h>

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

// WiFi credentials
const char* ssid = "Galaxy A53 5G225D";
const char* password = "sdci3924";

// MQTT Broker settings
const char* mqtt_broker = "10.150.80.246";  // Change to your MQTT broker IP
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_UGLL-E";

// MQTT Topics
const char* topic_motor = "/motor";

WiFiClient espClient;
PubSubClient mqtt_client(espClient);

unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 5000; // 5 seconds

void IRAM_ATTR isr() {
    buttonPressedFlag = true;
}

void setup_wifi() {
    delay(10);
    Serial.println();
    Serial.print("Connecting to WiFi");
    
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println();
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    digitalWrite(WIFI_LED, HIGH);
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message arrived [");
    Serial.print(topic);
    Serial.print("] ");
    
    // Convert payload to string
    String message = "";
    for (int i = 0; i < length; i++) {
        message += (char)payload[i];
    }
    Serial.println(message);
    
    if (strcmp(topic, topic_motor) == 0) {
        // Parse command_num from JSON or use raw string
        // Expected format: {"command_num": 1, ...}
        // For simplicity, extract command_num
        int cmdStart = message.indexOf("\"command_num\":");
        if (cmdStart != -1) {
            cmdStart += 14; // Skip to value
            int cmdEnd = message.indexOf(",", cmdStart);
            if (cmdEnd == -1) cmdEnd = message.indexOf("}", cmdStart);
            
            String cmdNum = message.substring(cmdStart, cmdEnd);
            cmdNum.trim();
            
            if (cmdNum != "" && cmdNum != currentCommand) {
                currentCommand = cmdNum;
                Serial.print("New command: ");
                Serial.println(currentCommand);
                lastCommandTime = millis();
            }
        }
    }
}

void mqtt_reconnect() {
    while (!mqtt_client.connected()) {
        Serial.print("Attempting MQTT connection...");
        
        if (mqtt_client.connect(mqtt_client_id)) {
            Serial.println("connected");
            // Subscribe to motor topic
            mqtt_client.subscribe(topic_motor);
            Serial.print("Subscribed to: ");
            Serial.println(topic_motor);
        } else {
            Serial.print("failed, rc=");
            Serial.print(mqtt_client.state());
            Serial.println(" try again in 5 seconds");
            delay(5000);
        }
    }
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

    digitalWrite(WIFI_LED, LOW);

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
    setup_wifi();

    // MQTT
    mqtt_client.setServer(mqtt_broker, mqtt_port);
    mqtt_client.setCallback(mqtt_callback);

    //set motor A and motor B speed, 0-255 255 being the fastest
    analogWrite(PWMA, motorSpeed);
    analogWrite(PWMB, motorSpeed);
    analogWrite(PWMC, motorSpeed);
    analogWrite(PWMD, motorSpeed - 20);
}

void loop() {
    currentTime = millis();

    // MQTT connection check
    if (!mqtt_client.connected()) {
        digitalWrite(WIFI_LED, LOW);
        mqtt_reconnect();
    }
    mqtt_client.loop();

    // WiFi status LED
    if (WiFi.status() == WL_CONNECTED && mqtt_client.connected()) {
        digitalWrite(WIFI_LED, HIGH);
    } else {
        digitalWrite(WIFI_LED, LOW);
        Serial.println("WiFi or MQTT Disconnected");
    }

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
        //BUZZ
        Serial.println("too far");
    }
    else if (command == TOO_NEAR) {
        offAllMotor();
        //BUZZ
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
        // delay(1000);
    } 
    else if (command == TURN_RIGHT) {
        moveForward(TIME);
        moveTurnRight(ROTATE_TIME);
        Serial.println("rotate right");
        // delay(1000);
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