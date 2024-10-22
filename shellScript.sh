#!/bin/bash

# Loop to check if Bluetooth PAN interface (bnep0) is up
while true; do
  if [[ $(ifconfig | grep bnep0) ]]; then
    echo "Bluetooth tethering detected, starting detection script..."
    # Run your Python script for alcohol and face detection
    python3 /home/pi/detection_script.py
    exit 0
  else
    echo "No Bluetooth tethering detected. Retrying in 5 seconds..."
  fi
  sleep 5
done
