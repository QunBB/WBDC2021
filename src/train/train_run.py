import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.client import device_lib
import yaml
import os
import time

import src.model.core as get_model
from src.model.core.match_tower import MatchTower
from src.model.core.optimization import create_optimizer
from src.prepare.input_builder import file_based_input_fn_builder
from src.train.utils import get_tf_hashtable, compute_weighted_score

from config.config import *

tf.random.set_random_seed(RANDOM_SEED)


def get_gpu_info(num_gpu):
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

    return num_towers, device_string


def build_train_graph(model, match_tower, inputs, init_lr, num_train_steps, num_warmup_steps, num_gpu):
    # GPU/CPU设置
    num_towers, device_string = get_gpu_info(num_gpu)

    # 将数据进行切分，放置到不同的GPU上
    tower_inputs = {key: tf.split(value, num_towers) for key, value in inputs.items()}
    tower_inputs = [{key: value[i] for key, value in tower_inputs.items()} for i in range(num_towers)]

    tower_loss_dict = {}
    tower_pred_dict = {}
    tower_label_dict = {}

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("match_tower", reuse=tf.AUTO_REUSE):
                    _, _, match_logits, _ = match_tower(tower_inputs[i], is_training=True)

                with tf.variable_scope("core_model", reuse=tf.AUTO_REUSE):
                    loss_dict, pred_dict, label_dict = model(tower_inputs[i], is_training=True,
                                                             match_logits=match_logits)

                concat_tensor_dict(tower_loss_dict, loss_dict)
                concat_tensor_dict(tower_pred_dict, pred_dict)
                concat_tensor_dict(tower_label_dict, label_dict)

    train_op = create_optimizer(tower_loss_dict['total_loss'], init_lr, num_train_steps, num_warmup_steps)

    for d in [tower_pred_dict, tower_label_dict]:
        for key in d.keys():
            d[key] = tf.concat(d[key], axis=0)

    for key in tower_loss_dict.keys():
        tower_loss_dict[key] = tf.reduce_mean(tower_loss_dict[key])

    return train_op, tower_loss_dict, tower_pred_dict, tower_label_dict


def build_eval_graph(model, match_tower, inputs, num_gpu):
    # GPU/CPU设置
    num_towers, device_string = get_gpu_info(num_gpu)

    # 将数据进行切分，放置到不同的GPU上
    tower_inputs = {key: tf.split(value, num_towers) for key, value in inputs.items()}
    tower_inputs = [{key: value[i] for key, value in tower_inputs.items()} for i in range(num_towers)]

    tower_loss_dict = {}
    tower_pred_dict = {}
    tower_label_dict = {}

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("match_tower", reuse=tf.AUTO_REUSE):
                    _, _, match_logits, _ = match_tower(tower_inputs[i], is_training=False)
                with tf.variable_scope("core_model", reuse=tf.AUTO_REUSE):
                    loss_dict, pred_dict, label_dict = model(tower_inputs[i], is_training=False, match_logits=match_logits)

                concat_tensor_dict(tower_loss_dict, loss_dict)
                concat_tensor_dict(tower_pred_dict, pred_dict)
                concat_tensor_dict(tower_label_dict, label_dict)

    for d in [tower_pred_dict, tower_label_dict]:
        for key in d.keys():
            d[key] = tf.concat(d[key], axis=0)

    for key in tower_loss_dict.keys():
        tower_loss_dict[key] = tf.reduce_mean(tower_loss_dict[key])

    return tower_loss_dict, tower_pred_dict, tower_label_dict


def compute_accuracy(pred_dict, label_dict, loss_weight=None):
    accuracy_dict = {}
    for name in pred_dict.keys():
        pred = pred_dict[name]
        label = label_dict[name]
        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])

        pred = (pred >= 0.5).astype(np.int32)
        accuracy_dict[name] = np.mean((pred == label).astype(np.float32))

    if loss_weight is not None:
        compute_weight_metrics(accuracy_dict, loss_weight)

    return accuracy_dict


def compute_recall(pred_dict, label_dict, loss_weight=None):
    recall_dict = {}
    for name in pred_dict.keys():
        pred = pred_dict[name]
        label = label_dict[name]

        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])

        positive_index = np.where(label == 1)

        if len(positive_index[0]) == 0:
            recall_dict[name] = 0.
            continue

        pred = pred[positive_index]
        label = label[positive_index]

        pred = (pred >= 0.5).astype(np.int32)
        recall_dict[name] = np.mean((pred == label).astype(np.float32))

    if loss_weight is not None:
        compute_weight_metrics(recall_dict, loss_weight)

    return recall_dict


def compute_uauc(pred_dict, label_dict, userid_arr, loss_weight=None):
    userid_arr = np.reshape(userid_arr, [-1])
    all_userid = set(userid_arr)

    for key in pred_dict.keys():
        pred_dict[key] = np.reshape(pred_dict[key], [-1])

    auc_dict = {}
    for name in pred_dict.keys():
        auc_dict[name] = []

        for uid in all_userid:
            index = np.where(userid_arr == uid)

            pred = pred_dict[name][index]
            label = label_dict[name][index]

            # 全是正样本或全是负样本，auc为nan，进行过滤
            if min(label) == max(label):
                continue

            fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auc_dict[name].append(auc)

        auc_dict[name] = sum(auc_dict[name]) / len(auc_dict[name])

    if loss_weight is not None:
        compute_weight_metrics(auc_dict, {k: v for k, v in loss_weight.items() if
                                          k in ['read_comment', 'like', 'click_avatar', 'forward',
                                                'comment', 'follow', 'favorite']})

    return auc_dict


def compute_weight_metrics(metrics_dict, loss_weight_dict):
    weight_sum = sum(loss_weight_dict.values())
    loss_weight_dict = {k: v / weight_sum for k, v in loss_weight_dict.items()}
    metrics_dict['weighted'] = sum([loss_weight_dict[name] * metrics_dict[name] for name in metrics_dict.keys()])


def concat_tensor_dict(tower_dict, tensor_dict):
    for key, values in tensor_dict.items():
        if key in tower_dict:
            tower_dict[key].append(values)
        else:
            tower_dict[key] = [values]


def get_features_name(csv):
    return [name for name in ['read_comment', 'like', 'click_avatar', 'forward', 'comment', 'follow', 'favorite'] if
            name in csv.columns]


def run(match_tower_config, model_name, model_config, optimizer_config, train_input_file,
        eval_input_file, hashtable_ckpt, match_tower_ckpt=None, init_checkpoint=None):
    config = tf.ConfigProto(
        allow_soft_placement=True,  # 如果你指定的设备不存在，自动分配设备
        log_device_placement=False  # 是否打印设备分配日志
    )
    config.gpu_options.allow_growth = True

    print("The match tower config:")
    print(match_tower_config)
    print("The core model config:")
    print(model_config)

    hashtable_list, hashtable, table = get_tf_hashtable(feed_num=match_tower_config['feed_vocab_dict']['feedid'])

    model_config.update({"tables": table, "hashtables": hashtable})

    match_tower_config.update({"tables": table, "hashtables": hashtable})

    match_tower = MatchTower(**match_tower_config)

    model = get_model.get_instance(model_name, model_config)

    # 创建模型保存目录
    os.system("mkdir -p {}".format(os.path.dirname(optimizer_config['model_path'])))

    train_inputs = file_based_input_fn_builder(tf.gfile.Glob(train_input_file + "/*.tfrecord"),
                                               model_config['max_seq_len_dict'],
                                               is_training=True,
                                               batch_size=optimizer_config['train_batch_size'])

    # 为了能够重复获取数据，设置is_training=True
    eval_inputs = file_based_input_fn_builder(eval_input_file,
                                              model_config['max_seq_len_dict'],
                                              is_training=True,
                                              batch_size=optimizer_config['eval_batch_size'])
    train_op, train_tower_loss_dict, train_tower_pred_dict, train_tower_label_dict = build_train_graph(
        model,
        match_tower,
        train_inputs,
        init_lr=optimizer_config['init_lr'],
        num_train_steps=optimizer_config['num_train_steps'],
        num_warmup_steps=optimizer_config['num_warmup_steps'],
        num_gpu=optimizer_config['num_gpu'])

    eval_tensor_loss_dict, eval_tensor_pred_dict, eval_tensor_label_dict = build_eval_graph(model, match_tower,
                                                                                            eval_inputs,
                                                                                            num_gpu=optimizer_config['num_gpu'])

    hashtable_saver = tf.train.Saver(var_list=hashtable_list)

    save_variables = tf.global_variables()
    print("************************ All Global Variables ************************")
    for v in save_variables:
        print(v)
    print("************************ Variables not to save ************************")
    for v in tf.global_variables("hashtable"):
        save_variables.remove(v)
        print(v)

    match_tower_vars = tf.global_variables("tower/match_tower") + tf.global_variables("share_embedding_table")
    match_tower_saver = tf.train.Saver(var_list=match_tower_vars)
    print("************************ Match Tower Variables ************************")
    for v in match_tower_vars:
        print(v)

    saver = tf.train.Saver(var_list=save_variables, max_to_keep=10)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        hashtable_saver.restore(sess, hashtable_ckpt)

        if match_tower_ckpt is not None:
            match_tower_saver.restore(sess, tf.train.latest_checkpoint(match_tower_ckpt))

        if init_checkpoint is not None:
            saver.restore(sess, init_checkpoint)

        for name in hashtable.keys():
            size = sess.run(hashtable[name].size())
            assert size > 0, "HashTable: {} load with failure".format(name)

        start_step = sess.run(tf.train.get_or_create_global_step())
        sess.run(tf.assign(tf.train.get_or_create_global_step(), 0))
        print("The start_step is {}, global step is: {}".format(start_step,
                                                                sess.run(tf.train.get_global_step())))

        start_time = time.time()
        best_recall, best_auc = -1, -1
        for i in range(start_step, start_step+optimizer_config['num_train_steps']):
            _, train_loss_dict, train_pred_dict, train_label_dict = sess.run(
                [train_op, train_tower_loss_dict, train_tower_pred_dict, train_tower_label_dict])

            if (i + 1) % optimizer_config['train_log_steps'] == 0:
                print("========================= global_step: {} ========================= ".format(i))
                print("loss: {}".format(train_loss_dict))

                train_accuracy_dict = compute_accuracy(train_pred_dict, train_label_dict,
                                                       loss_weight=model_config['loss_weight'])
                train_recall_dict = compute_recall(train_pred_dict, train_label_dict,
                                                   loss_weight=model_config['loss_weight'])
                print("accuracy: {}".format(train_accuracy_dict))
                print("recall: {}".format(train_recall_dict))

            if (i + 1) % optimizer_config['run_eval_steps'] == 0 or i == optimizer_config['num_train_steps'] - 1:
                print("********* Runing Eval *********")
                eval_loss_dict, eval_pred_dict, eval_label_dict = {}, {}, {}
                userid_list = []
                for _ in range(optimizer_config['steps_every_eval']):
                    batch_eval_loss_dict, batch_eval_pred_dict, batch_eval_label_dict, userid = sess.run(
                        [eval_tensor_loss_dict, eval_tensor_pred_dict, eval_tensor_label_dict,
                         eval_inputs['userid_origin']])
                    userid_list.append(userid)
                    for batch_dict, eval_dict in zip(
                            [batch_eval_loss_dict, batch_eval_pred_dict, batch_eval_label_dict],
                            [eval_loss_dict, eval_pred_dict, eval_label_dict]):
                        for key in batch_dict.keys():
                            if key in eval_dict:
                                eval_dict[key].append(batch_dict[key])
                            else:
                                eval_dict[key] = [batch_dict[key]]
                for d in [eval_loss_dict, eval_pred_dict, eval_label_dict]:
                    for key in d.keys():
                        if d == eval_loss_dict:
                            d[key] = np.mean(d[key])
                        else:
                            d[key] = np.concatenate(d[key], axis=0)

                # 计算加权的auc
                eval_result_dict = {"result_" + name: res for name, res in eval_pred_dict.items()}
                eval_result_dict.update({"label_" + name: res for name, res in eval_label_dict.items()})
                eval_result_dict.update({"userid": np.concatenate(userid_list, axis=0)})
                # print({name: len(arr) for name, arr in eval_result_dict.items()})
                eval_df = pd.DataFrame(eval_result_dict)
                del eval_result_dict, eval_pred_dict, eval_label_dict, userid_list
                eval_auc = compute_weighted_score(eval_df, model_config['loss_weight'])

                # eval_auc = compute_uauc(eval_pred_dict, eval_label_dict, np.concatenate(userid_list, axis=0),
                #                         loss_weight=model_config['loss_weight'])
                # eval_recall = compute_recall(eval_pred_dict, eval_label_dict, loss_weight=model_config['loss_weight'])
                print("eval_loss: {}".format(eval_loss_dict))
                # print("eval_accuracy: {}".format(
                #     compute_accuracy(eval_pred_dict, eval_label_dict, loss_weight=model_config['loss_weight'])))
                # print("eval_recall: {}".format(eval_recall))
                print("eval_auc: {}".format(eval_auc))
                print("********* End Eval *********")

                if (i + 1) % optimizer_config['save_model_steps'] == 0 or i == optimizer_config['num_train_steps'] - 1:
                    print("+++++++++++ cost time: {} /minute +++++++++++".format((time.time() - start_time) // 60))
                    start_time = time.time()
                    # 只有aAuc或召回率提升时，才进行模型的保存
                    if eval_auc['weighted'] > best_auc:
                        saver.save(sess, "%s_%.6f-%d" % (optimizer_config['model_path'], eval_auc['weighted'], i),
                                   write_meta_graph=False)
                        best_auc = eval_auc['weighted']
                        # best_recall = eval_recall['weighted']


if __name__ == '__main__':
    tf.logging.set_verbosity("INFO")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config.yaml', type=str)
    parser.add_argument('--match_config_file', default='config.yaml', type=str)
    parser.add_argument('--train_input_file', default='feed_emb', type=str)
    parser.add_argument('--eval_input_file', default='feed_emb', type=str)
    parser.add_argument('--hashtable_ckpt', default=None, type=str)
    parser.add_argument('--match_tower_ckpt', default=None, type=str)
    parser.add_argument('--init_checkpoint', default=None, type=str)

    args = parser.parse_args()

    config = yaml.load(open(args.config_file, 'r'))
    match_config = yaml.load(open(args.match_config_file, 'r'))

    run(match_tower_config=match_config['ModelConfig'],
        model_name=config['ModelName'],
        model_config=config['ModelConfig'],
        optimizer_config=config['OptimizerConfig'],
        train_input_file=args.train_input_file,
        eval_input_file=args.eval_input_file,
        hashtable_ckpt=args.hashtable_ckpt,
        match_tower_ckpt=args.match_tower_ckpt,
        init_checkpoint=args.init_checkpoint)
