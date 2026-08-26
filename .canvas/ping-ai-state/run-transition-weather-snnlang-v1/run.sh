#!/bin/bash
tools/snnsim/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ai.bundle --transition-bundle .canvas/ping-ai-state/snnlang/ping.bundle --transition-start-ms 500 --transition-end-ms 1000 --t-ms 1500 --seed 7 --out-dir .canvas/ping-ai-state/run-transition-weather-snnlang-v1 --wipe-dir
