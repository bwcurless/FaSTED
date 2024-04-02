#!/usr/bin/env bash
#
# -f overwrite profile if it exists
ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 4" --clock-control=none --set full "release/main" /data/expo/expo_16D_2000000.txt 0.1 4
ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 41" --clock-control=none --set full "release/main" /data/expo/expo_16D_2000000.txt 0.1 41
ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 42" --clock-control=none --set full "release/main" /data/expo/expo_16D_2000000.txt 0.1 42
ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 43" --clock-control=none --set full "release/main" /data/expo/expo_16D_2000000.txt 0.1 43
