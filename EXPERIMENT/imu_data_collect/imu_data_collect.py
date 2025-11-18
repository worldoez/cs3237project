import serial as pyserial
import csv
from datetime import datetime
import sys

# Configuration
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
CSV_FILENAME = f'imu_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def main():
    if len(sys.argv) < 2:
        print("Usage: python imu_data_collect.py ACTION_LABEL")
        sys.exit(1)

    action_label = sys.argv[1]
    print(f"Action Label: {action_label}")
    print(f"Connecting to {SERIAL_PORT}...")

    try:
        ser = pyserial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected!")
        print(f"Data will be saved to: {CSV_FILENAME}\n")

        csv_file = open(CSV_FILENAME, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        print("Waiting for data...")
        print("-" * 50)

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if line.startswith("HEADER:"):
                    header = line.replace("HEADER:", "")
                    header_fields = header.split(',')
                    csv_writer.writerow(header_fields + ['action_label'])
                    csv_file.flush()
                    print(f"✓ CSV Header: {header},action_label")

                elif line.startswith("DATA:"):
                    data = line.replace("DATA:", "")
                    if csv_writer:
                        csv_writer.writerow(data.split(',') + [action_label])
                        csv_file.flush()
                        print(f"📊 {data} {action_label}")

                elif line.startswith("STATUS:"):
                    print(f"⚡ {line.replace('STATUS:', '')}")

                elif line.startswith("INIT:"):
                    print(f"🔧 {line.replace('INIT:', '')}")

                elif line.startswith("ERROR:"):
                    print(f"❌ {line.replace('ERROR:', '')}")

                elif line.startswith("READY:"):
                    print(f"✅ {line.replace('READY:', '')}")
                    print("-" * 50)

    except pyserial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}")
        print(f"Details: {e}")
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  - {port.device}: {port.description}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nStopping data collection...")
        csv_file.close()
        ser.close()
        print(f"Data saved to: {CSV_FILENAME}")
        print("Goodbye!")

    except Exception as e:
        print(f"Unexpected error: {e}")
        csv_file.close()
        ser.close()
        sys.exit(1)

if __name__ == "__main__":
    main()