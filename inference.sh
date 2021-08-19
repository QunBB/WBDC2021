#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

source activate /home/tione/notebook/envs/tf_1

python -u src/inference.py \
--test_csv data/wedata/wechat_algo_data2/test_a.csv \
--config_file config/config.yaml \
--match_config_file config/match_tower.yaml \
--hashtable_ckpt data/preprocess/tf_hash_test/user_feed.ckpt \
--init_checkpoint data/model/ple_model.ckpt_0.764256-346689 \
--result_path data/submission/result.csv