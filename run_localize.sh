#!/bin/bash

source .venv/bin/activate

python src/localize.py --input reference_samples/ --out outputs/json/ --crops outputs/crops/
