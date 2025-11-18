#include <WiFi.h>
#include <WebServer.h>

const char* ap_ssid = "UGLEE-WIFI";
const char* ap_password = "05112025";

WebServer server(80); // HTTP server on port 80

void handleRoot() {
    server.send(200, 
        "text/html", 
        "<h1> Hello from ESP32 DevKit! </h1>
        <p> It works </p>"
    );
}

void handleData() {
    if (server.hasArg("msg")) {
        String msg = server.arg("msg");
        Serial.print("Received message: ");
        Serial.println(msg);
        server.send(200, "text/plain", "ESP32 received: " + msg);
    } else {
        server.send(400, "text/plain", "Missing 'msg' parameter");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("Starting AP...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(ap_ssid, ap_password);
    Serial.print("AP IP addr: ");
    Serial.println(WiFi.softAPIP());

    // Define routes
    server.on("/", handleRoot);
    server.on("/send", handleData);
    server.begin();

    Serial.println("HTTP server started");
    Serial.print("Connect to this WiFi: ");
    Serial.println(ap_ssid);
    Serial.println("Then open http://192.168.4.1 in your browser");
}

void loop() {
  server.handleClient(); // process incoming http requests
}
