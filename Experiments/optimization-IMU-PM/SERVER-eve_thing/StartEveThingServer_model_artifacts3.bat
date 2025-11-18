@echo off
set PYTHONPATH=%cd%
python src/realtime/http_ingest_server.py ^
  --artifacts models_artifacts3 ^
  --window 12 ^
  --threshold 0.45 ^
  --threshold_left 0.00 --threshold_right 0.55 --threshold_straight 0.45 --threshold_jump 0.72 ^
  --debounce 1 --debounce_straight 0 --debounce_jump 2 ^
  --turn_fast_thr 0.36 --turn_min_gyro 0.03 ^
  --quiet_gyro 0.09 --quiet_ms 80 ^
  --jump_gate 16.0 --jump_hold_ms 180 ^
  --ema_alpha 0.56 ^
  --yaw_prior_k 2.8 --yaw_prior_min 0.04 --yaw_prior_max 0.20 ^
  --auto_calib 1 ^
  --host 0.0.0.0 --http_port 5000
pause