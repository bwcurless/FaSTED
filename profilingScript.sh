#!/usr/bin/env bash
#
# -replay-mode=application, so it doesn't replay kernels individually. I believe there is not enough global memory on the GPU, we hang. It tries to just replay certain kernels over and over, but you get bad results if you do that because some kernels can't be repeated without rerunning all of them. application option means the entire application will be restarted when you profile, this guarantees there will be no errors while executing, but is slower to profile.
# -f overwrite profile if it exists
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 4" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 4
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "mode 41" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 41
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 42" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 42
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 43" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 43
ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 44" --clock-control=none --set full "release/main" /data/expo/expo_16D_262144.txt 0.1 44
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 45" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 45
#ncu -f  --app-replay-buffer=file --replay-mode=application -o "Mode 46" --clock-control=none --set full "release/main" ~/datasets/expo_16D_200000.txt 0.1 46
