#!/bin/bash

poetry install

git clone https://github.com/czarobxm/fast-transformers.git
cd fast-transformers
srun --partition=common --qos=1gpu4h --gres=gpu:1 poetry run python3 setup.py install
cd ..
rm -r fast-transformers

poetry update