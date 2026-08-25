#!/bin/bash
tools/snn/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ai.bundle --input synthetic-spikes --t-ms 10 --seed 7 --input-rate 1 --independent-drive 8 2.10 --independent-drive-i 8 0.25 --out-dir .canvas/ping-ai-state/bundle-smoke --wipe-dir
