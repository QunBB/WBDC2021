import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.python.client import device_lib
import traceback
import time
from tqdm import tqdm

import src.model.core as get_model
from config.config import *
from src.model.core.match_tower import MatchTower
from src.train.utils import get_tf_hashtable

tf.random.set_random_seed(RANDOM_SEED)


def build_net_graph(model, match_tower, inputs, num_gpu):
    # GPU/CPU设置
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'][:num_gpu]
    num_gpus = len(gpus)
    if num_gpus > 0:
        tf.logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        tf.logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'
    num_towers = num_towers
    device_string = device_string

    # assert num_gpus == len(tower_inputs)

    # 将数据进行切分，放置到不同的GPU上
    tower_inputs = {key: tf.split(value, num_towers) for key, value in inputs.items()}
    tower_inputs = [{key: value[i] for key, value in tower_inputs.items()} for i in range(num_towers)]

    tower_pred_dict = {}

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("match_tower", reuse=tf.AUTO_REUSE):
                    _, _, match_logits, _ = match_tower(tower_inputs[i], is_training=False)

                with tf.variable_scope("core_model", reuse=tf.AUTO_REUSE):
                    _, pred_dict, _ = model(tower_inputs[i], is_training=False,
                                            match_logits=match_logits)

                concat_tensor_dict(tower_pred_dict, pred_dict)

    for key in tower_pred_dict.keys():
        tower_pred_dict[key] = tf.concat(tower_pred_dict[key], axis=0)

    return tower_pred_dict


def concat_tensor_dict(tower_dict, tensor_dict):
    for key, values in tensor_dict.items():
        if key in tower_dict:
            tower_dict[key].append(values)
        else:
            tower_dict[key] = [values]


def input_fn_builder(data, features_name, batch_size):
    d = tf.data.Dataset.from_tensor_slices({name: data[name].values for name in features_name})

    d = d.batch(batch_size, drop_remainder=False)

    iters = d.make_one_shot_iterator()
    batch = iters.get_next()

    for name in list(batch.keys()):
        t = batch[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)

        if t.dtype == tf.float64:
            t = tf.cast(t, tf.float32)
        batch[name] = t

    return batch


def run_one_model(data, match_tower_config, model_name, model_config, optimizer_config,
                  hashtable_ckpt, init_checkpoint):
    tf.reset_default_graph()

    test_size = len(data)

    config = tf.ConfigProto(
        allow_soft_placement=True,  # 如果你指定的设备不存在，自动分配设备
        log_device_placement=False  # 是否打印设备分配日志
    )
    config.gpu_options.allow_growth = True

    print("The match tower config:")
    print(match_tower_config)
    print("The core model config:")
    print(model_config)

    hashtable_list, hashtable, table = get_tf_hashtable(feed_num=match_tower_config['feed_vocab_dict']['feedid'],
                                                        user_num=match_tower_config['user_vocab_dict']['userid'])

    model_config.update({"tables": table, "hashtables": hashtable})

    match_tower_config.update({"tables": table, "hashtables": hashtable})

    match_tower = MatchTower(**match_tower_config)

    model = get_model.get_instance(model_name, model_config)

    steps = test_size // optimizer_config['test_batch_size']
    print("train parallel: {}, residual: {}".format(steps * optimizer_config['test_batch_size'],
                                                    test_size - steps * optimizer_config['test_batch_size']))

    predict_inputs = input_fn_builder(
        data[['userid_origin', 'feedid_origin', 'device'] + LABELS_NAME][:steps * optimizer_config['test_batch_size']],
        features_name=['userid_origin', 'feedid_origin', 'device'] + LABELS_NAME,
        batch_size=optimizer_config['test_batch_size'])

    tower_pred_dict = build_net_graph(
        model,
        match_tower,
        predict_inputs,
        num_gpu=optimizer_config['num_gpu'])

    # 由于tf.split将数据分配给gpu时，需要完全可整除，剩余的不可整除部分由一个gpu处理
    residual_inputs = None
    if steps * optimizer_config['test_batch_size'] < test_size:
        residual_inputs = input_fn_builder(
            data[['userid_origin', 'feedid_origin', 'device'] + LABELS_NAME][
            steps * optimizer_config['test_batch_size']:],
            features_name=['userid_origin', 'feedid_origin', 'device'] + LABELS_NAME,
            batch_size=optimizer_config['test_batch_size'] // optimizer_config['num_gpu'])
        residual_pred_dict = build_net_graph(
            model,
            match_tower,
            residual_inputs,
            num_gpu=1)

    hashtable_saver = tf.train.Saver(var_list=hashtable_list)

    save_variables = tf.global_variables()
    print("************************ All Global Variables ************************")
    for v in save_variables:
        print(v)
    print("************************ Variables not to load ************************")
    for v in tf.global_variables("hashtable"):
        save_variables.remove(v)
        print(v)

    saver = tf.train.Saver(var_list=save_variables, max_to_keep=10)

    result = {}
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        hashtable_saver.restore(sess, hashtable_ckpt)

        saver.restore(sess, init_checkpoint)

        for name in hashtable.keys():
            size = sess.run(hashtable[name].size())
            assert size > 0, "HashTable: {} load with failure".format(name)

        for _ in tqdm(range(steps), total=steps):
            predict_one_batch(sess=sess,
                              predict_inputs=predict_inputs,
                              tower_pred_dict=tower_pred_dict,
                              result=result)

        print("finished num: {}".format(sum([len(v) for v in result['userid']])))

        if residual_inputs is not None:
            while True:
                try:
                    predict_one_batch(sess=sess,
                                      predict_inputs=residual_inputs,
                                      tower_pred_dict=residual_pred_dict,
                                      result=result)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("!!!! test run end !!!!")
                    break

    for key in result.keys():
        result[key] = np.concatenate(result[key], axis=0)

    pred_len = list(map(lambda x: len(x), result.values()))
    assert min(pred_len) == max(pred_len)
    result_df = pd.DataFrame(result)

    return result_df[['userid', 'feedid'] + LABELS_NAME]


def run(test_csv, match_tower_config, model_name, model_config, optimizer_config,
        hashtable_ckpt, result_path, init_checkpoint):
    assert len(match_tower_config) == len(model_config) == len(optimizer_config) == len(init_checkpoint)

    start_time = time.time()

    data = pd.read_csv(test_csv)

    test_size = len(data)

    data['feedid_origin'] = data['feedid']
    data['userid_origin'] = data['userid']
    data['device'] = data['device'] - 1
    for name in LABELS_NAME:
        data[name] = 0
    data.index = range(test_size)

    result_df = None
    for one_match_tower_config, one_model_config, one_optimizer_config, one_init_checkpoint, one_model_name in zip(
            match_tower_config,
            model_config,
            optimizer_config,
            init_checkpoint,
            model_name):
        one_result = run_one_model(data, match_tower_config=one_match_tower_config,
                                   model_config=one_model_config,
                                   optimizer_config=one_optimizer_config,
                                   init_checkpoint=one_init_checkpoint,
                                   model_name=one_model_name,
                                   hashtable_ckpt=hashtable_ckpt)
        if result_df is None:
            result_df = one_result
        else:
            result_df = pd.merge(result_df, one_result, on=['userid', 'feedid'])

            for name in LABELS_NAME:
                result_df[name] = result_df[name + "_x"] + result_df[name + "_y"]

            result_df = result_df[['userid', 'feedid'] + LABELS_NAME]

    for name in LABELS_NAME:
        result_df[name] = result_df[name] / len(init_checkpoint)

    # 概率保留6位小数
    for name in LABELS_NAME:
        result_df[name] = result_df[name].apply(lambda x: "{:.6f}".format(x))
    result_df[['userid', 'feedid'] + LABELS_NAME].to_csv(result_path,
                                                         encoding='utf-8',
                                                         index=False)
    result_size = len(result_df)
    print("+++++++++++ cost time: {} /minute +++++++++++".format((time.time() - start_time) // 60))
    print("+++++++++++ avg time: {} /ms +++++++++++".format((time.time() - start_time) * 1000 / test_size / 7 * 2000))

    assert test_size == result_size, "test_size: {}, result_size: {}".format(test_size, result_size)
    assert len(data[['userid', 'feedid']].drop_duplicates()) == len(result_df[['userid', 'feedid']].drop_duplicates())
    assert len(pd.merge(data, result_df, on=['userid', 'feedid'])) == test_size


def predict_one_batch(sess, predict_inputs, tower_pred_dict, result):
    userid, feedid, pred_dict = sess.run(
        [predict_inputs['userid_origin'],
         predict_inputs['feedid_origin'],
         tower_pred_dict])
    pred_dict.update({"userid": userid, "feedid": feedid})
    for key in pred_dict.keys():
        if key in result:
            result[key].append(pred_dict[key])
        else:
            result[key] = [pred_dict[key]]


if __name__ == '__main__':
    tf.logging.set_verbosity("INFO")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', default='config.yaml', type=str)
    parser.add_argument('--config_file', default='config.yaml', type=str)
    parser.add_argument('--match_config_file', default='config.yaml', type=str)
    parser.add_argument('--hashtable_ckpt', default=None, type=str)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--result_path', default=None, type=str)

    args = parser.parse_args()

    config = [yaml.load(open(config_file, 'r')) for config_file in args.config_file.split(",")]
    match_config = [yaml.load(open(match_config_file, 'r')) for match_config_file in args.match_config_file.split(",")]

    run(test_csv=args.test_csv,
        match_tower_config=[config['ModelConfig'] for config in match_config],
        model_name=[conf['ModelName'] for conf in config],
        model_config=[conf['ModelConfig'] for conf in config],
        optimizer_config=[conf['OptimizerConfig'] for conf in config],
        hashtable_ckpt=args.hashtable_ckpt,
        init_checkpoint=args.init_checkpoint.split(","),
        result_path=args.result_path)
