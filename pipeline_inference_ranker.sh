#!/bin/bash
set -e
python src/create_covisits_gpu.py local_validation=False
# python src/train_word2vec_model.py
# python src/create_candidates_df.py
python src/create_user_item_features.py local_validation=False
# python src/combine_candidates_user_item_features.py local_validation=False
papermill test.ipynb output.ipynb
