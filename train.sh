#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

source activate /home/tione/notebook/envs/tf_1

# 生成Map ID文件
python -u src/prepare/map_id_process.py \
--action_csv data/wedata/wechat_algo_data1/user_action.csv,data/wedata/wechat_algo_data2/user_action.csv \
--feed_csv data/wedata/wechat_algo_data2/feed_info.csv \
--save_path data/preprocess/map.obj

# DeepWalk
python -u src/prepare/deepwalk.py \
--action_csv data/wedata/wechat_algo_data1/user_action.csv,data/wedata/wechat_algo_data2/user_action.csv \
--test_csv data/wedata/wechat_algo_data1/test_a.csv,data/wedata/wechat_algo_data1/test_b.csv,data/wedata/wechat_algo_data2/test_a.csv \
--feed_csv data/wedata/wechat_algo_data2/feed_info.csv \
--save_dir data/deepwalk \
--map_file data/preprocess/map.obj \
--config_file config/deepwalk.yaml

# 文本Word2Vec
python -u src/prepare/nlp_process.py \
--feed_csv data/wedata/wechat_algo_data2/feed_info.csv \
--word_dim 512 \
--workers 50 \
--map_path data/preprocess/map.obj \
--save_path data/deepwalk/word.npy

# 生成tensorflow hashtable
python -u src/prepare/feed_hashtable.py \
--feed_csv data/wedata/wechat_algo_data2/feed_info.csv \
--map_file data/preprocess/map.obj \
--save_path data/preprocess/tf_hash/feed.ckpt

# 生成训练集tfrecord，带验证集
python -u src/prepare/multiprocessing_gene_dataset.py \
--csv_file data/wedata/wechat_algo_data1/user_action.csv,data/wedata/wechat_algo_data2/user_action.csv \
--feed_csv_file data/wedata/wechat_algo_data2/feed_info.csv \
--maps_file data/preprocess/map.obj \
--dataset_dir data/preprocess \
--pool_num 20 \
--do_eval true

# 生成训练集tfrecord，全量训练
python -u src/prepare/multiprocessing_gene_dataset.py \
--csv_file data/wedata/wechat_algo_data1/user_action.csv,data/wedata/wechat_algo_data2/user_action.csv \
--feed_csv_file data/wedata/wechat_algo_data2/feed_info.csv \
--maps_file data/preprocess/map.obj \
--dataset_dir data/preprocess \
--pool_num 20 \
--do_eval false

# 生成测试集的tensorflow hashtable
python -u src/prepare/gene_test_hashtable.py \
--feed_csv_file data/wedata/wechat_algo_data2/feed_info.csv \
--features_file data/preprocess/data_and_count.pkl \
--maps_file data/preprocess/map.obj \
--pool_num 40 \
--train_sess_path data/preprocess/tf_hash/feed.ckpt \
--save_path data/preprocess/tf_hash_test/user_feed.ckpt

# 训练Match Tower
python -u src/train/match_tower_run.py \
--action_csv data/wedata/wechat_algo_data1/user_action.csv,data/wedata/wechat_algo_data2/user_action.csv \
--feed_csv data/wedata/wechat_algo_data2/feed_info.csv \
--maps_file data/preprocess/map.obj \
--hashtable_ckpt data/preprocess/tf_hash/feed.ckpt \
--config_file config/match_tower.yaml \
--tfrecord_dir data/match_tower \
--full_run True

# 训练多任务模型
python -u src/train/train_run.py \
--config_file config/config.yaml \
--match_config_file config/match_tower.yaml \
--train_input_file data/preprocess/tfrecord/all_in/no_kfold/1 \
--eval_input_file data/preprocess/tfrecord/eval.tfrecord \
--hashtable_ckpt data/preprocess/tf_hash/feed.ckpt \
--match_tower_ckpt data/model/match_tower