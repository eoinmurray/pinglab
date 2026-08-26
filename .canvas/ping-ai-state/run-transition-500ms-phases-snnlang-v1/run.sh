#!/bin/bash
tools/snnsim/tool.py sim --bundle .canvas/ping-ai-state/snnlang/ai.bundle --transition-bundle .canvas/ping-ai-state/snnlang/ping.bundle --transition-start-ms 500 --transition-end-ms 1000 --input synthetic-spikes --t-ms 1500 --seed 7 --input-rate 1 --independent-drive 5 2.10 --independent-drive-i 5 0.25 --out-dir .canvas/ping-ai-state/run-transition-500ms-phases-snnlang-v1 --wipe-dir
