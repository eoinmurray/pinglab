#!/bin/bash
tools/snn/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ai.bundle --transition-bundle .canvas/ping-ai-state/snnlang/ping.bundle --transition-start-ms 20 --transition-end-ms 50 --input synthetic-spikes --t-ms 100 --seed 7 --input-rate 1 --independent-drive 8 2.10 --independent-drive-i 8 0.25 --out-dir .canvas/ping-ai-state/transition-smoke --wipe-dir
