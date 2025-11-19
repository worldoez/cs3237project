#!/bin/bash
export PYTHONPATH=$(pwd)
python3 src/realtime/http_ingest_server.py \
  --artifacts models_artifacts4 \
  --window 0 \
  --threshold 0.6 \
  --threshold_left 0.67 --threshold_right 0.55 --threshold_straight 0.53 --threshold_jump 0.60 \
  --debounce 3 --debounce_straight 2 --debounce_jump 2 \
  --turn_fast_thr 0.90 --turn_min_gyro 0.10 \
  --quiet_gyro 0.06 --quiet_ms 250 \
  --jump_gate 10.0 --jump_hold_ms 140 \
  --ema_alpha 0.6 \
  --yaw_prior_k 2.0 --yaw_prior_min 0.08 --yaw_prior_max 0.35 \
  --auto_calib 0 \
  --host 0.0.0.0 --http_port 5000