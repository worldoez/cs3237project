import paho.mqtt.client as mqtt
import json
from datetime import datetime
import csv
import sys
import signal

# Configuration
DATAPOINTS_PER_ACTION = 200
BROKER_IP = "10.180.115.152"
BROKER_PORT = 1883

# Global variables
action_id = 0
current_datapoints = []
is_recording = False
datapoint_count = 0
csv_filename = ""
action_label = ""
all_data = []

def on_connect(client, userdata, flags, rc):
    """Callback when connected to MQTT broker"""
    print(f"Connected to MQTT broker with result code: {rc}")
    
    # Subscribe to ESP32 topics
    client.subscribe("eve/imu")
    client.subscribe("eve/button")
    
    print("Subscribed to topics:")
    print("  - eve/imu (IMU sensor data)")
    print("  - eve/button (button events)")
    print("\nWaiting for ESP32 button press...\n")
    print("=" * 80)

def on_message(client, userdata, message):
    """Handle incoming MQTT messages"""
    global action_id, current_datapoints, is_recording, datapoint_count, all_data
    
    try:
        topic = message.topic
        payload = message.payload.decode('utf-8').strip()
        
        if topic == "eve/button":
            if payload == "start_recording":
                # Start new recording session
                action_id += 1
                datapoint_count = 0
                current_datapoints = []
                is_recording = True
                
                print(f"\nBegin Data Collection - action_id: {action_id} - action_label: {action_label}")
                print("-" * 80)
                
        elif topic == "eve/imu" and is_recording:
            # Parse JSON IMU data
            imu_data = json.loads(payload)
            
            datapoint_count += 1
            
            # Format: action_id, timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, action_label
            data_row = [
                action_id,
                imu_data['timestamp'],
                imu_data['gx'],
                imu_data['gy'],
                imu_data['gz'],
                imu_data['ax'],
                imu_data['ay'],
                imu_data['az'],
                action_label
            ]
            
            current_datapoints.append(data_row)
            all_data.append(data_row)
            
            # Print the data point
            print(f"{datapoint_count}: {action_id}, {imu_data['timestamp']}, "
                  f"{imu_data['gx']:.4f}, {imu_data['gy']:.4f}, {imu_data['gz']:.4f}, "
                  f"{imu_data['ax']:.4f}, {imu_data['ay']:.4f}, {imu_data['az']:.4f}, "
                  f"{action_label}")
            
            # Check if we've collected enough datapoints
            if datapoint_count >= DATAPOINTS_PER_ACTION:
                is_recording = False
                print("-" * 80)
                print(f"End Data Collection - action_id: {action_id} - action_label: {action_label}")
                print(f"Collected {datapoint_count} datapoints")
                print("=" * 80)
                print("\nReady for next recording. Press button on ESP32...\n")
                
                # Send stop command to ESP32
                client.publish("eve/control", "stop_recording", 0, False)
    
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {topic}: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")

def save_to_csv():
    """Save all collected data to CSV file"""
    if not all_data:
        print("No data to save.")
        return
    
    print(f"\nSaving data to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow([
            'action_id', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z',
            'accel_x', 'accel_y', 'accel_z', 'action_label'
        ])
        
        # Write all data
        csv_writer.writerows(all_data)
    
    print(f"✓ Saved {len(all_data)} datapoints to {csv_filename}")
    print(f"✓ Total actions recorded: {action_id}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global client
    print("\n\n" + "=" * 80)
    print("Stopping data collection...")
    print("=" * 80)
    
    save_to_csv()
    
    client.loop_stop()
    client.disconnect()
    print("Goodbye!")
    sys.exit(0)

def main():
    global csv_filename, action_label, client
    
    if len(sys.argv) < 2:
        print("Usage: python mqtt_imu_collect.py ACTION_LABEL")
        sys.exit(1)
    
    action_label = sys.argv[1]
    csv_filename = f'imu_data_{action_label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    print("=" * 80)
    print("ESP32 MQTT IMU Data Collector")
    print("=" * 80)
    print(f"Action Label: {action_label}")
    print(f"Datapoints per action: {DATAPOINTS_PER_ACTION}")
    print(f"Data will be saved to: {csv_filename}")
    print("=" * 80)
    
    # Set up MQTT client
    client = mqtt.Client(client_id="imu_collector_python")
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\nConnecting to MQTT broker at {BROKER_IP}:{BROKER_PORT}...")
    
    try:
        client.connect(BROKER_IP, BROKER_PORT, 60)
        
        # Start MQTT loop
        client.loop_start()
        
        # Keep the program running
        while True:
            pass
    
    except Exception as e:
        print(f"Error: {e}")
        client.loop_stop()
        client.disconnect()
        sys.exit(1)

if __name__ == "__main__":
    main()