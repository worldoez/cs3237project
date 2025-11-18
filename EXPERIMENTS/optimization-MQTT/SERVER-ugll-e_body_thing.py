# import paho.mqtt.client as mqtt
# import json
# import time
# import psycopg2
# from psycopg2 import pool
# from contextlib import contextmanager

# # MQTT Configuration
# MQTT_BROKER = "localhost"  # Change to your broker address
# MQTT_PORT = 1883
# MQTT_KEEPALIVE = 60

# # Topics
# TOPIC_MOTOR = "/motor"
# TOPIC_OBSTACLE = "/obstacle"
# TOPIC_IMU_OUT = "/imu_out"
# TOPIC_CAM_OUT = "/cam_out"

# # Database pool
# db_pool = psycopg2.pool.SimpleConnectionPool(
#     1,
#     10,
#     database="postgres",
#     user="postgres",
#     host="localhost",
#     password="11223344",
#     port=5431,
# )

# @contextmanager
# def get_db_connection():
#     conn = db_pool.getconn()
#     try:
#         yield conn
#         conn.commit()
#     except Exception:
#         conn.rollback()
#         raise
#     finally:
#         db_pool.putconn(conn)

# class MotorController:
#     def __init__(self):
#         self.client = mqtt.Client()
#         self.client.on_connect = self.on_connect
#         self.client.on_message = self.on_message
        
#         # Store latest data from sensors
#         self.latest_distance = -1
#         self.latest_command = "STOP"
#         self.data_received = {"obstacle": False, "imu": False}
        
#     def on_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             print("Connected to MQTT Broker!")
#             # Subscribe to sensor topics
#             client.subscribe(TOPIC_OBSTACLE)
#             client.subscribe(TOPIC_IMU_OUT)
#             client.subscribe(TOPIC_CAM_OUT)
#             print(f"Subscribed to: {TOPIC_OBSTACLE}, {TOPIC_IMU_OUT}, {TOPIC_CAM_OUT}")
#         else:
#             print(f"Failed to connect, return code {rc}")
    
#     def on_message(self, client, userdata, msg):
#         try:
#             payload = json.loads(msg.payload.decode())
            
#             if msg.topic == TOPIC_OBSTACLE:
#                 # Handle obstacle/distance data from ESP32
#                 self.latest_distance = float(payload.get("distance", -1))
#                 self.data_received["obstacle"] = True
#                 print(f"Received distance: {self.latest_distance} cm")
                
#             elif msg.topic == TOPIC_IMU_OUT:
#                 # Handle IMU/pose command data
#                 self.latest_command = payload.get("command", "STOP")
#                 self.data_received["imu"] = True
#                 print(f"Received command: {self.latest_command}")
                
#             elif msg.topic == TOPIC_CAM_OUT:
#                 # Handle camera/ML data if needed
#                 print(f"Received camera data: {payload}")
            
#             # Process and send motor command when we have data
#             if self.data_received["obstacle"] and self.data_received["imu"]:
#                 self.process_and_publish_motor_command()
                
#         except json.JSONDecodeError as e:
#             print(f"JSON decode error: {e}")
#         except Exception as e:
#             print(f"Error processing message: {e}")
    
#     def process_and_publish_motor_command(self):
#         """Process sensor data and publish motor command"""
#         distance = self.latest_distance
#         command = self.latest_command
        
#         # Optional: Store in database
#         # try:
#         #     with get_db_connection() as conn:
#         #         with conn.cursor() as cur:
#         #             cur.execute(
#         #                 "INSERT INTO test_table (distance, command) VALUES (%s, %s)",
#         #                 (distance, command),
#         #             )
#         # except Exception as e:
#         #     print(f"Database error: {e}")
        
#         # Determine motor command number based on logic
#         commandNum = self.determine_command_number(distance, command)
        
#         # Publish to motor topic
#         motor_payload = {
#             "command_num": commandNum,
#             "command_name": command,
#             "distance": distance,
#             "timestamp": time.time()
#         }
        
#         self.client.publish(TOPIC_MOTOR, json.dumps(motor_payload), qos=1)
#         print(f"Published to {TOPIC_MOTOR}: {motor_payload}")
        
#         # Reset flags to wait for next update
#         self.data_received = {"obstacle": False, "imu": False}
    
#     def determine_command_number(self, distance, command):
#         """Convert command and distance to command number"""
#         # Priority order: JUMP > distance checks > movement commands
#         if command == "JUMP":
#             return 8
#         elif distance == -1:
#             return 0  # No valid distance
#         elif distance >= 55:
#             return 9  # Too far
#         elif distance <= 25:
#             return 10  # Too near
#         elif command == "STRAIGHT":
#             return 1
#         elif command == "BACKWARD":
#             return 2
#         elif command == "LEFT":
#             return 3
#         elif command == "RIGHT":
#             return 4
#         elif command == "SIDE_LEFT":
#             return 5
#         elif command == "SIDE_RIGHT":
#             return 6
#         elif command == "FULL_TURN":
#             return 7
#         else:
#             return 0  # STOP or unknown
    
#     def run(self):
#         """Connect and start the MQTT loop"""
#         try:
#             self.client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
#             print(f"Connecting to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
#             self.client.loop_forever()
#         except KeyboardInterrupt:
#             print("\nShutting down...")
#             self.client.disconnect()
#         except Exception as e:
#             print(f"Error: {e}")

# if __name__ == "__main__":
#     controller = MotorController()
#     controller.run()

import paho.mqtt.client as mqtt
import json
import time
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import argparse

class MotorController:
    def __init__(self, broker_ip, broker_port, db_config, enable_db=False):
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.enable_db = enable_db
        
        # MQTT Topics
        self.TOPIC_MOTOR = "/motor"
        self.TOPIC_OBSTACLE = "/obstacle"
        self.TOPIC_IMU_OUT = "/imu_out"
        self.TOPIC_CAM_OUT = "/cam_out"
        
        # Initialize database pool if enabled
        self.db_pool = None
        if enable_db:
            try:
                self.db_pool = psycopg2.pool.SimpleConnectionPool(
                    1, 10,
                    database=db_config['database'],
                    user=db_config['user'],
                    host=db_config['host'],
                    password=db_config['password'],
                    port=db_config['port']
                )
                print(f"Database pool initialized: {db_config['host']}:{db_config['port']}")
            except Exception as e:
                print(f"Failed to initialize database pool: {e}")
                self.enable_db = False
        
        # MQTT Client setup
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Store latest data from sensors
        self.latest_distance = -1
        self.latest_command = "STOP"
        self.data_received = {"obstacle": False, "imu": False}
        
    @contextmanager
    def get_db_connection(self):
        if not self.enable_db or not self.db_pool:
            raise Exception("Database not enabled")
        conn = self.db_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.db_pool.putconn(conn)
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            # Subscribe to sensor topics
            client.subscribe(self.TOPIC_OBSTACLE)
            client.subscribe(self.TOPIC_IMU_OUT)
            client.subscribe(self.TOPIC_CAM_OUT)
            print(f"Subscribed to: {self.TOPIC_OBSTACLE}, {self.TOPIC_IMU_OUT}, {self.TOPIC_CAM_OUT}")
        else:
            print(f"Failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            if msg.topic == self.TOPIC_OBSTACLE:
                # Handle obstacle/distance data from ESP32
                self.latest_distance = float(payload.get("distance", -1))
                self.data_received["obstacle"] = True
                print(f"Received distance: {self.latest_distance} cm")
                
            elif msg.topic == self.TOPIC_IMU_OUT:
                # Handle IMU/pose command data
                self.latest_command = payload.get("command", "STOP")
                self.data_received["imu"] = True
                print(f"Received command: {self.latest_command}")
                
            elif msg.topic == self.TOPIC_CAM_OUT:
                # Handle camera/ML data if needed
                print(f"Received camera data: {payload}")
            
            # Process and send motor command when we have data
            if self.data_received["obstacle"] and self.data_received["imu"]:
                self.process_and_publish_motor_command()
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def process_and_publish_motor_command(self):
        """Process sensor data and publish motor command"""
        distance = self.latest_distance
        command = self.latest_command
        
        # Optional: Store in database if enabled
        if self.enable_db:
            try:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO test_table (distance, command) VALUES (%s, %s)",
                            (distance, command),
                        )
                print("Data stored in database")
            except Exception as e:
                print(f"Database error: {e}")
        
        # Determine motor command number based on logic
        commandNum = self.determine_command_number(distance, command)
        
        # Publish to motor topic
        motor_payload = {
            "command_num": commandNum,
            "command_name": command,
            "distance": distance,
            "timestamp": time.time()
        }
        
        self.client.publish(self.TOPIC_MOTOR, json.dumps(motor_payload), qos=1)
        print(f"Published to {self.TOPIC_MOTOR}: {motor_payload}")
        
        # Reset flags to wait for next update
        self.data_received = {"obstacle": False, "imu": False}
    
    def determine_command_number(self, distance, command):
        """Convert command and distance to command number"""
        # Priority order: JUMP > distance checks > movement commands
        if command == "JUMP":
            return 8
        elif distance == -1:
            return 0  # No valid distance
        elif distance >= 55:
            return 9  # Too far
        elif distance <= 25:
            return 10  # Too near
        elif command == "STRAIGHT":
            return 1
        elif command == "BACKWARD":
            return 2
        elif command == "LEFT":
            return 3
        elif command == "RIGHT":
            return 4
        elif command == "SIDE_LEFT":
            return 5
        elif command == "SIDE_RIGHT":
            return 6
        elif command == "FULL_TURN":
            return 7
        else:
            return 0  # STOP or unknown
    
    def run(self):
        """Connect and start the MQTT loop"""
        try:
            self.client.connect(self.broker_ip, self.broker_port, 60)
            print(f"Connecting to MQTT Broker at {self.broker_ip}:{self.broker_port}")
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.client.disconnect()
            if self.db_pool:
                self.db_pool.closeall()
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='MQTT Motor Controller')
    
    # MQTT arguments
    parser.add_argument('--broker', '-b', default='localhost',
                        help='MQTT broker IP address (default: localhost)')
    parser.add_argument('--port', '-p', type=int, default=1883,
                        help='MQTT broker port (default: 1883)')
    
    # Database arguments
    parser.add_argument('--enable-db', action='store_true',
                        help='Enable database logging')
    parser.add_argument('--db-host', default='localhost',
                        help='Database host (default: localhost)')
    parser.add_argument('--db-port', type=int, default=5431,
                        help='Database port (default: 5431)')
    parser.add_argument('--db-name', default='postgres',
                        help='Database name (default: postgres)')
    parser.add_argument('--db-user', default='postgres',
                        help='Database user (default: postgres)')
    parser.add_argument('--db-password', default='11223344',
                        help='Database password (default: 11223344)')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    print("=" * 50)
    print("MQTT Motor Controller Starting")
    print("=" * 50)
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Database: {'Enabled' if args.enable_db else 'Disabled'}")
    if args.enable_db:
        print(f"DB Host: {args.db_host}:{args.db_port}")
    print("=" * 50)
    
    controller = MotorController(
        broker_ip=args.broker,
        broker_port=args.port,
        db_config=db_config,
        enable_db=args.enable_db
    )
    controller.run()

if __name__ == "__main__":
    main()