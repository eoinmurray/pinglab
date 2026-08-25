#!/bin/bash
/Users/eoin/pinglab/tools/snn/tool.py sim --model ping --input synthetic-spikes --n-in 100 --n-hidden 100 --n-inh 25 --private-w-in --input-rate 60 --t-ms 100 --dt 0.25 --n-batch 1 --seed 7 --outputs rasters --out-dir .canvas/ping-emergence/run --wipe-dir
