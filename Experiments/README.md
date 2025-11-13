Network / flow after changes (how it works):

All devices (ESP32-CAM, ESP32 motor board, the laptop/server(s) running IMU server + aggregator + DB) join the same Wi‑Fi hotspot (your external hotspot/router). Each device gets a LAN IP on that subnet.
Camera joins hotspot in STA mode and publishes distance to flask_server_distance (on host IP:5001).
IMU server posts predictions to its endpoint (host:5000).
Aggregator (flask_server_to_bot on host) polls /getDistance and /control on the LAN addresses (or localhost) and returns combined JSON on /fetchData.
Motor (ESP32) polls aggregator /fetchData and applies the command; it ignores stale commands via CMD_STALE_MS.
Important: Flask apps must run bound to 0.0.0.0 (they do in your files) and firewall on host must allow ports (5000, 5001, 5002).


PS:
- If the ESP32-CAM must use a static IP, use WiFi.config(...) in CameraWebServer.ino (example in file commented). Otherwise DHCP from hotspot is fine.


Quick test checklist (do in order):

Set hotspot SSID/password in CameraWebServer.ino, upload to ESP32-CAM. Watch Serial, note camera IP.
Start flask_server_distance.py on host (port 5001).
Start IMU server (port 5000) and confirm /control returns JSON (use curl).
Update flask_server_to_bot.py DIST_URL/IMU_URL if needed, start aggregator (port 5002).
On host, curl aggregator: curl http://<AGGREGATOR_IP>:5002/fetchData and verify JSON with distance, command, timestamps.
Update motor_test.ino serverName to aggregator IP, upload to motor ESP32, start it.
Watch Serial on motor: it should print "CTRL OK" lines and react. Move IMU / camera to simulate changes.
If camera fails, aggregator should still return IMU values; motor should either use last fresh command or go to failsafe.

Extra tips if network weak:

Increase HTTP timeouts in sketches (http.setTimeout), but keep aggregator short so it is resilient.
Reduce motor poll rate (POLL_MS or loop delay) to reduce Wi‑Fi load.
If still unreliable, run a local MQTT broker (Mosquitto) on the host and switch devices to publish/subscribe — more robust under packet loss.