#!/bin/bash

source .venv/bin/activate

#python src/classify.py --input reference_samples/ --out outputs/json/

python src/classify.py --input reference_samples/ --out outputs/json/ --model qwen2.5vl:7b