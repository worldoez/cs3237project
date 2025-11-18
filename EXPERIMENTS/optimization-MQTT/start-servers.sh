#!/bin/bash

# ============================================
# Configuration Variables
# ============================================

# Broker Configuration
BROKER_IP="10.180.115.152"
BROKER_PORT=1883

# Database Configuration
DB_HOST="localhost"
DB_PORT=5431
DB_NAME="postgres"
DB_USER="postgres"
DB_PASSWORD="11223344"

# HTTP Ingest Server Configuration
HTTP_HOST="0.0.0.0"
HTTP_PORT=5000
ARTIFACTS_PATH="./SERVER-eve_thing/models_artifacts"

# Model Thresholds
THRESHOLD=0.45
THRESHOLD_LEFT=0.38
THRESHOLD_RIGHT=0.52
THRESHOLD_STRAIGHT=0.32
THRESHOLD_JUMP=0.55

# Timing Parameters
WINDOW=12
DEBOUNCE=1
DEBOUNCE_STRAIGHT=1
DEBOUNCE_JUMP=1
QUIET_MS=120
JUMP_HOLD_MS=100

# Turn Parameters
TURN_FAST_THR=0.50
TURN_MIN_GYRO=0.08

# Other Parameters
QUIET_GYRO=0.09
JUMP_GATE=12.0
EMA_ALPHA=0.74
YAW_PRIOR_K=1.8
YAW_PRIOR_MIN=0.14
YAW_PRIOR_MAX=0.25
AUTO_CALIB=0

# Camera Server Configuration
CAMERA_BROKER="192.168.1.100"
CAMERA_INTERVAL=0.1

# ============================================
# Start Servers
# ============================================

export PYTHONPATH=$(pwd)

# Start HTTP Ingest Server
python3 ./SERVER-eve_thing/src/realtime/http_ingest_server.py \
  --artifacts "$ARTIFACTS_PATH" \
  --window $WINDOW \
  --threshold $THRESHOLD \
  --threshold_left $THRESHOLD_LEFT \
  --threshold_right $THRESHOLD_RIGHT \
  --threshold_straight $THRESHOLD_STRAIGHT \
  --threshold_jump $THRESHOLD_JUMP \
  --debounce $DEBOUNCE \
  --debounce_straight $DEBOUNCE_STRAIGHT \
  --debounce_jump $DEBOUNCE_JUMP \
  --turn_fast_thr $TURN_FAST_THR \
  --turn_min_gyro $TURN_MIN_GYRO \
  --quiet_gyro $QUIET_GYRO \
  --quiet_ms $QUIET_MS \
  --jump_gate $JUMP_GATE \
  --jump_hold_ms $JUMP_HOLD_MS \
  --ema_alpha $EMA_ALPHA \
  --yaw_prior_k $YAW_PRIOR_K \
  --yaw_prior_min $YAW_PRIOR_MIN \
  --yaw_prior_max $YAW_PRIOR_MAX \
  --auto_calib $AUTO_CALIB \
  --broker_ip $BROKER_IP \
  --broker_port $BROKER_PORT & \

# Start Motor Controller
python3 SERVER-ugll-e_body_thing.py \
  --broker $BROKER_IP \
  --port $BROKER_PORT \
  --enable-db \
  --db-host $DB_HOST \
  --db-port $DB_PORT \
  --db-name $DB_NAME \
  --db-user $DB_USER \
  --db-password $DB_PASSWORD &

# Start Camera Server
python3 SERVER-ugll-e_eye_thing.py \
  --broker $CAMERA_BROKER \
  --port $BROKER_PORT \
  --interval $CAMERA_INTERVAL &

# Wait for all background processes
wait

echo "All servers stopped."
