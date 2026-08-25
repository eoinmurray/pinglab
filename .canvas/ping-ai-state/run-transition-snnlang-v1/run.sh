#!/bin/bash
tools/snn/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ai.bundle --transition-bundle .canvas/ping-ai-state/snnlang/ping.bundle --transition-start-ms 1000 --transition-end-ms 2000 --input synthetic-spikes --t-ms 3600 --seed 7 --input-rate 1 --independent-drive 8 2.10 --independent-drive-i 8 0.25 --out-dir .canvas/ping-ai-state/run-transition-snnlang-v1 --wipe-dir
