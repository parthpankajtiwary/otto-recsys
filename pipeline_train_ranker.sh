#!/bin/bash
set -e
python src/create_covisits_gpu.py
# python src/train_word2vec_model.py
# python src/create_candidates_df.py
python src/create_user_item_features.py
# python src/combine_candidates_user_item_features.py
# python src/add_targets_to_candidate_df.py
papermill test.ipynb output.ipynb
