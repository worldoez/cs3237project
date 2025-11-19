// DIDN'T WORK
#include <WiFi.h>
#include <HTTPClient.h>

// ===== WiFi + server =====
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";
const char* SERVER    = "http://laptop_ip:5000";  
const char* DEVICE_ID = "imu01";

// ===== Enable a one-time self test at boot (spins A,B,C,D sequentially) =====
const bool RUN_SELF_TEST = true;   // set false after verifying wiring
const int  SELF_TEST_MS  = 1000;   // per-motor spin time

// ===== Wiring from motor.ino labels =====
// A -> Back Left
// B -> Back Right
// C -> Front Right
// D -> Front Left

// Standby pins for two TB6612 boards
const int STBY1 = 26;  // board #1 STBY
const int STBY2 = 27;  // board #2 STBY

// Motor A (Back Left) on board #1
const int PWMA = 5;   // speed
const int AIN1 = 17;  // dir
const int AIN2 = 16;  // dir

// Motor B (Back Right) on board #2
const int PWMB = 14;
const int BIN1 = 13;
const int BIN2 = 12;

// Motor C (Front Right) on board #2
const int PWMC = 21;
const int CIN1 = 18;
const int CIN2 = 19;

// Motor D (Front Left) on board #1
const int PWMD = 15;
const int DIN1 = 0;
const int DIN2 = 2;

// Map to sides for differential control:
// LEFT side = D (front-left) + A (rear-left)
// RIGHT side = C (front-right) + B (rear-right)

// ===== Control loop params =====
const unsigned long POLL_MS     = 50;    // ~20 Hz polling
const unsigned long FAILSAFE_MS = 600;   // stop if stale
const int PWM_FREQ = 20000;              // 20 kHz
const int PWM_RES  = 8;                  // 8-bit
const float SLEW_MAX = 0.12f;            // max step per update in [-1..1]

// Per-wheel sign flips (use to correct direction in STRAIGHT)
float SIGN_FL = 1.0f;  // D (Front Left)
float SIGN_RL = 1.0f;  // A (Back Left)
float SIGN_FR = 1.0f;  // C (Front Right)
float SIGN_RR = 1.0f;  // B (Back Right)

// ===== Implementation =====
struct MotorChan {
  int in1, in2, pin, ch;
  float lastCmd;
  float sign;
};

// Construct with LEDC channels
MotorChan FL = {DIN1, DIN2, PWMD, 0, 0.0f, 1.0f}; // D -> Front Left
MotorChan RL = {AIN1, AIN2, PWMA, 1, 0.0f, 1.0f}; // A -> Back Left
MotorChan FR = {CIN1, CIN2, PWMC, 2, 0.0f, 1.0f}; // C -> Front Right
MotorChan RR = {BIN1, BIN2, PWMB, 3, 0.0f, 1.0f}; // B -> Back Right

float targetLeft = 0.0f, targetRight = 0.0f;
unsigned long lastPoll = 0, lastOK = 0;

static inline float clampf(float v, float lo, float hi){ return v<lo?lo:(v>hi?hi:v); }
static inline int toPwm(float x) { x = clampf(x, 0.0f, 1.0f); return (int)(x * ((1<<PWM_RES)-1) + 0.5f); }

void standby(bool on) {
  digitalWrite(STBY1, on ? HIGH : LOW);
  digitalWrite(STBY2, on ? HIGH : LOW);
}

void setupMotor(MotorChan &m) {
  pinMode(m.in1, OUTPUT); pinMode(m.in2, OUTPUT);
  pinMode(m.pin, OUTPUT);
  ledcSetup(m.ch, PWM_FREQ, PWM_RES);
  ledcAttachPin(m.pin, m.ch);
}

void applyMotor(MotorChan &m, float cmd) {
  cmd *= m.sign; // per‑wheel sign
  float dir = (cmd >= 0.0f) ? 1.0f : -1.0f;
  float mag = fabsf(cmd);
  if (dir >= 0) { digitalWrite(m.in1, HIGH); digitalWrite(m.in2, LOW); }
  else          { digitalWrite(m.in1, LOW);  digitalWrite(m.in2, HIGH); }
  ledcWrite(m.ch, toPwm(mag)); // channel write
}

float slew(float current, float target) {
  float d = target - current;
  if (d > SLEW_MAX) d = SLEW_MAX;
  if (d < -SLEW_MAX) d = -SLEW_MAX;
  return current + d;
}

void setTargets(float left, float right) {
  targetLeft  = clampf(left,  -1.0f, 1.0f);
  targetRight = clampf(right, -1.0f, 1.0f);
}

bool fetchControl(float &left, float &right) {
  String url = String(SERVER) + "/control?device_id=" + DEVICE_ID;
  HTTPClient http; http.setTimeout(1500); http.begin(url);
  int code = http.GET();
  if (code != 200) { http.end(); return false; }
  String body = http.getString(); http.end();
  auto parseKey = [&](const char* key, float &out)->bool{
    int k = body.indexOf(key); if (k < 0) return false;
    k = body.indexOf(':', k); if (k < 0) return false;
    int j = k+1; while (j < body.length() && (body[j]==' ')) j++;
    int e = j; while (e < body.length() && (isdigit(body[e]) || body[e]=='-' || body[e]=='.' || body[e]=='e')) e++;
    out = body.substring(j, e).toFloat(); return true;
  };
  float l=0, r=0;
  if (!parseKey("left_speed", l) || !parseKey("right_speed", r)) {
    // fallback: use "speed" for both if provided
    float sp=0;
    if (!parseKey("speed", sp)) return false;
    l = r = sp;
  }
  left = l; right = r; return true;
}

void offAll() {
  digitalWrite(FL.in1, LOW); digitalWrite(FL.in2, LOW); ledcWrite(FL.ch, 0);
  digitalWrite(RL.in1, LOW); digitalWrite(RL.in2, LOW); ledcWrite(RL.ch, 0);
  digitalWrite(FR.in1, LOW); digitalWrite(FR.in2, LOW); ledcWrite(FR.ch, 0);
  digitalWrite(RR.in1, LOW); digitalWrite(RR.in2, LOW); ledcWrite(RR.ch, 0);
}

void selfTest() {
  Serial.println("SelfTest: A(BL), B(BR), C(FR), D(FL) each forward...");
  standby(true);
  // A -> Back Left (RL)
  applyMotor(RL, +0.6f); delay(SELF_TEST_MS); offAll(); delay(400);
  // B -> Back Right (RR)
  applyMotor(RR, +0.6f); delay(SELF_TEST_MS); offAll(); delay(400);
  // C -> Front Right (FR)
  applyMotor(FR, +0.6f); delay(SELF_TEST_MS); offAll(); delay(400);
  // D -> Front Left (FL)
  applyMotor(FL, +0.6f); delay(SELF_TEST_MS); offAll(); delay(400);
  standby(false);
}

void setup() {
  Serial.begin(115200); delay(200);

  // STBY pins
  pinMode(STBY1, OUTPUT);
  pinMode(STBY2, OUTPUT);
  standby(false);

  // Motors and LEDC channels
  setupMotor(FL); setupMotor(RL); setupMotor(FR); setupMotor(RR);

  // Per-wheel direction sign (tune if a wheel goes backward in STRAIGHT)
  FL.sign = SIGN_FL; RL.sign = SIGN_RL; FR.sign = SIGN_FR; RR.sign = SIGN_RR;

  // Wi‑Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi connecting");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - t0) < 15000) { delay(250); Serial.print("."); }
  Serial.println(WiFi.status()==WL_CONNECTED? " OK":" FAIL");

  if (RUN_SELF_TEST) selfTest();

  lastPoll = millis(); lastOK = millis();
  standby(true); // keep enabled
}

void loop() {
  unsigned long now = millis();

  // Poll /control
  if (now - lastPoll >= POLL_MS) {
    lastPoll = now;
    if (WiFi.status() == WL_CONNECTED) {
      float l, r;
      if (fetchControl(l, r)) {
        lastOK = now;
        setTargets(l, r);
      }
    }
  }

  // Failsafe: stop if stale
  if (now - lastOK > FAILSAFE_MS) setTargets(0, 0);

  // Slew per wheel toward targets
  FL.lastCmd = slew(FL.lastCmd, targetLeft);
  RL.lastCmd = slew(RL.lastCmd, targetLeft);
  FR.lastCmd = slew(FR.lastCmd, targetRight);
  RR.lastCmd = slew(RR.lastCmd, targetRight);

  // Apply to drivers
  applyMotor(FL, FL.lastCmd);
  applyMotor(RL, RL.lastCmd);
  applyMotor(FR, FR.lastCmd);
  applyMotor(RR, RR.lastCmd);

  delay(5);
}