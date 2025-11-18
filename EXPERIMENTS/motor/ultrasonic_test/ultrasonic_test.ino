// Ultrasonic Sensor Test for ESP32
// TRIG: GPIO 35
// ECHO: GPIO 34

const int PIN_TRIG = 4;
const int PIN_ECHO = 35;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n========================================");
  Serial.println("ESP32 Ultrasonic Sensor Test");
  Serial.println("========================================");
  Serial.printf("TRIG Pin: GPIO %d\n", PIN_TRIG);
  Serial.printf("ECHO Pin: GPIO %d\n", PIN_ECHO);
  Serial.println("========================================\n");
  
  // Configure pins
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  
  // Initial state
  digitalWrite(PIN_TRIG, LOW);
  delay(100);
  
  Serial.println("Setup complete. Starting measurements...\n");
}

void loop() {
  Serial.println("--- New Reading ---");
  
  // Test 1: Check pin states
  Serial.print("1. Initial TRIG state: ");
  Serial.println(digitalRead(PIN_TRIG) ? "HIGH" : "LOW");
  
  Serial.print("2. Initial ECHO state: ");
  Serial.println(digitalRead(PIN_ECHO) ? "HIGH" : "LOW");
  
  // Test 2: Toggle TRIG manually
  Serial.println("3. Toggling TRIG pin...");
  digitalWrite(PIN_TRIG, HIGH);
  delay(10);
  digitalWrite(PIN_TRIG, LOW);
  Serial.println("   TRIG toggled");
  
  // Test 3: Send ultrasonic pulse
  Serial.println("4. Sending ultrasonic pulse...");
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  
  // Test 4: Wait for echo with different timeouts
  Serial.println("5. Waiting for ECHO response...");
  
  // Try short timeout first
  unsigned long duration1 = pulseIn(PIN_ECHO, HIGH, 10000UL); // 10ms
  Serial.printf("   10ms timeout - Duration: %lu µs\n", duration1);
  
  delay(100);
  
  // Send pulse again
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  
  // Try longer timeout
  unsigned long duration2 = pulseIn(PIN_ECHO, HIGH, 50000UL); // 50ms
  Serial.printf("   50ms timeout - Duration: %lu µs\n", duration2);
  
  delay(100);
  
  // Send pulse again
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  
  // Try very long timeout
  unsigned long duration3 = pulseIn(PIN_ECHO, HIGH, 100000UL); // 100ms
  Serial.printf("   100ms timeout - Duration: %lu µs\n", duration3);
  
  // Test 5: Calculate distances if any reading succeeded
  Serial.println("\n6. Results:");
  
  if (duration1 > 0) {
    float distance1 = (duration1 * 0.0343) / 2.0;
    Serial.printf("   ✓ Distance (10ms): %.2f cm\n", distance1);
  } else {
    Serial.println("   ✗ No response with 10ms timeout");
  }
  
  if (duration2 > 0) {
    float distance2 = (duration2 * 0.0343) / 2.0;
    Serial.printf("   ✓ Distance (50ms): %.2f cm\n", distance2);
  } else {
    Serial.println("   ✗ No response with 50ms timeout");
  }
  
  if (duration3 > 0) {
    float distance3 = (duration3 * 0.0343) / 2.0;
    Serial.printf("   ✓ Distance (100ms): %.2f cm\n", distance3);
  } else {
    Serial.println("   ✗ No response with 100ms timeout");
  }
  
  // Test 6: Check if ECHO ever goes HIGH
  Serial.println("\n7. Monitoring ECHO pin for 1 second...");
  unsigned long startTime = millis();
  bool echoWentHigh = false;
  
  // Send one more pulse
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  
  while (millis() - startTime < 1000) {
    if (digitalRead(PIN_ECHO) == HIGH) {
      echoWentHigh = true;
      Serial.println("   ✓ ECHO went HIGH!");
      break;
    }
  }
  
  if (!echoWentHigh) {
    Serial.println("   ✗ ECHO never went HIGH in 1 second");
    Serial.println("\n⚠️  PROBLEM DETECTED:");
    Serial.println("   - ECHO pin is not responding");
    Serial.println("   - Check: Wiring, Power (5V), Sensor damage");
    Serial.println("   - Note: GPIO 35 is INPUT-ONLY on ESP32!");
    Serial.println("           It CANNOT output signals for TRIG");
  }
  
  Serial.println("\n========================================");
  Serial.println("Waiting 3 seconds before next test...\n");
  delay(3000);
}
