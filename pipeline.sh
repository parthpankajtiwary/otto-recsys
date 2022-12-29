#!/bin/bash
set -e
# python src/create_covisits_gpu.py
# python src/train_word2vec_model.py
# python src/create_user_item_features.py
python src/inference.py
