#!/bin/sh
uv run python experiments/exp078.py --stage calibrate
uv run python experiments/exp078.py --stage sweep
