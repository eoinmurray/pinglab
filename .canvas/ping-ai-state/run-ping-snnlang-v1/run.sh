#!/bin/bash
tools/snn/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ping.bundle --input synthetic-spikes --t-ms 1000 --seed 7 --input-rate 1 --independent-drive 8 2.10 --independent-drive-i 8 0.25 --out-dir .canvas/ping-ai-state/run-ping-snnlang-v1 --wipe-dir
