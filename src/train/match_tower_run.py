import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.python.client import device_lib
import yaml
import os
import pickle
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
from tqdm import tqdm

from src.model.core.match_tower import MatchTower
from src.model.core.optimization import create_optimizer
from src.train.utils import get_tf_hashtable

from config.config import *

tf.random.set_random_seed(RANDOM_SEED)


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # 需要注意这里接受的格式是list，并且只能是一维的
    return f


def build_train_graph(model, inputs, init_lr, num_train_steps, num_warmup_steps, num_gpu):
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

    # 将数据进行切分，放置到不同的GPU上
    tower_inputs = {key: tf.split(value, num_towers) for key, value in inputs.items()}
    tower_inputs = [{key: value[i] for key, value in tower_inputs.items()} for i in range(num_towers)]

    tower_loss = []
    tower_pred = []
    tower_label = []

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("match_tower", reuse=tf.AUTO_REUSE):
                    loss, pred, _, label = model(tower_inputs[i], is_training=True)
                    tower_loss.append(loss)
                    tower_pred.append(pred)
                    tower_label.append(label)

    train_op = create_optimizer(tower_loss, init_lr, num_train_steps, num_warmup_steps)

    tower_pred = tf.concat(tower_pred, axis=0)
    tower_label = tf.concat(tower_label, axis=0)

    tower_loss = tf.reduce_mean(tower_loss)

    return train_op, tower_loss, tower_pred, tower_label


def build_eval_graph(model, inputs):
    with tf.variable_scope("tower", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("match_tower", reuse=tf.AUTO_REUSE):
            loss, pred, _, label = model(inputs, is_training=False)

    return loss, pred, label


def write_tfrecord(action_csv, feed_csv, map_file, tfrecord_dir):
    os.system("mkdir -p {}".format(tfrecord_dir))

    data = pd.DataFrame()
    for csv in action_csv.split(","):
        data = data.append(pd.read_csv(csv))

    feed_info = pd.read_csv(feed_csv)
    with open(map_file, "rb") as file:
        mappings = pickle.load(file)

    # 毫秒转化为秒
    data['play'] = data['play'] / 1000
    data['stay'] = data['stay'] / 1000

    data['score'] = data['read_comment'] + data['comment'] + data['like'] + data['click_avatar'] + data['forward'] + \
                    data['follow'] + data['favorite']

    data = pd.merge(data, feed_info, on='feedid')

    data['play_rate'] = data['play'] / data['videoplayseconds']
    data['stay_rate'] = data['stay'] / data['videoplayseconds']

    data['userid'] = data['userid'].apply(lambda x: mappings['userid'][x])
    data['feedid_origin'] = data['feedid']
    data['device'] = data['device'] - 1
    data['labels'] = (data['play_rate'] >= 0.7) | (data['stay'] >= 60) | (data['score'] > 0)
    data['labels'] = data['labels'].astype(int)

    print("All data num: {}, positive num: {}".format(len(data), len(data.loc[data['labels'] == 1])))

    data = data.sample(frac=1)

    eval_num = int(len(data) * 0.1)

    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, "eval.tfrecord"))

    print("Write into eval tfrecord......")
    for userid, feedid_origin, device, labels in tqdm(
            data[['userid', 'feedid_origin', 'device', 'labels']].values[:eval_num], total=eval_num):
        features = OrderedDict()
        features['userid'] = create_int_feature([userid])
        features['feedid_origin'] = create_int_feature([feedid_origin])
        features['device'] = create_int_feature([device])
        features['labels'] = create_int_feature([labels])
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())

    writer.close()

    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, "train.tfrecord"))

    print("Write into train tfrecord......")
    for userid, feedid_origin, device, labels in tqdm(
            data[['userid', 'feedid_origin', 'device', 'labels']].values[eval_num:], total=len(data) - eval_num):
        features = OrderedDict()
        features['userid'] = create_int_feature([userid])
        features['feedid_origin'] = create_int_feature([feedid_origin])
        features['device'] = create_int_feature([device])
        features['labels'] = create_int_feature([labels])
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())

    writer.close()


def input_fn_builder(input_file, batch_size, is_training):
    name_to_features = {
        "userid": tf.FixedLenFeature([], tf.int64),
        "feedid_origin": tf.FixedLenFeature([], tf.int64),
        "device": tf.FixedLenFeature([], tf.int64),
        "labels": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=False))

    iters = d.make_one_shot_iterator()
    batch = iters.get_next()

    return batch


def run(tfrecord_dir, action_csv, feed_csv, map_file, model_config, optimizer_config, hashtable_ckpt,
        full_run=False, init_checkpoint=None):
    if not (os.path.exists(os.path.join(tfrecord_dir, "train.tfrecord")) and os.path.exists(
            os.path.join(tfrecord_dir, "eval.tfrecord"))):
        write_tfrecord(action_csv, feed_csv, map_file, tfrecord_dir)

    config = tf.ConfigProto(
        allow_soft_placement=True,  # 如果你指定的设备不存在，自动分配设备
        log_device_placement=False  # 是否打印设备分配日志
    )
    config.gpu_options.allow_growth = True

    print(model_config)

    hashtable_list, hashtable, table = get_tf_hashtable(feed_num=model_config['feed_vocab_dict']['feedid'])

    model_config.update({"tables": table, "hashtables": hashtable})

    model = MatchTower(**model_config)

    # 创建模型保存目录
    os.system("mkdir -p {}".format(os.path.dirname(optimizer_config['model_path'])))

    train_file = [os.path.join(tfrecord_dir, "train.tfrecord")]
    if full_run:
        train_file.append(os.path.join(tfrecord_dir, "eval.tfrecord"))
        tf.logging.warn("************ Match tower Full train ************")

    train_inputs = input_fn_builder(train_file,
                                    batch_size=optimizer_config['train_batch_size'],
                                    is_training=True)

    eval_inputs = input_fn_builder(os.path.join(tfrecord_dir, "eval.tfrecord"),
                                   batch_size=optimizer_config['eval_batch_size'],
                                   is_training=True)

    train_op, train_tower_loss, train_tower_pred, train_tower_label = build_train_graph(
        model,
        train_inputs,
        init_lr=optimizer_config['init_lr'],
        num_train_steps=optimizer_config['num_train_steps'],
        num_warmup_steps=optimizer_config['num_warmup_steps'],
        num_gpu=optimizer_config['num_gpu'])

    eval_tensor_loss, eval_tensor_pred, eval_tensor_label = build_eval_graph(model, eval_inputs)

    hashtable_saver = tf.train.Saver(var_list=hashtable_list)

    save_variables = tf.global_variables()
    print("All Global Variables:")
    for v in save_variables:
        print(v)
    print("Variables not to save:")
    for v in tf.global_variables("hashtable"):
        save_variables.remove(v)
        print(v)

    saver = tf.train.Saver(var_list=save_variables, max_to_keep=10)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        hashtable_saver.restore(sess, hashtable_ckpt)

        for name in hashtable.keys():
            size = sess.run(hashtable[name].size())
            assert size > 0, "HashTable: {} load with failure".format(name)

        if init_checkpoint is not None:
            saver.restore(sess, init_checkpoint)

        best_score = -1
        for i in range(optimizer_config['num_train_steps']):
            _, train_loss, train_pred, train_label = sess.run(
                [train_op, train_tower_loss, train_tower_pred, train_tower_label])

            if (i + 1) % optimizer_config['train_log_steps'] == 0:
                print("========================= global_step: {} ========================= ".format(i))
                print("loss: {}, auc: {}, recall: {}".format(train_loss,
                                                             accuracy_score(train_label.astype(int),
                                                                            (train_pred > 0.5).astype(int)),
                                                             recall_score(train_label.astype(int),
                                                                          (train_pred > 0.5).astype(int))))

            if (i + 1) % optimizer_config['run_eval_steps'] == 0 or i == optimizer_config['num_train_steps'] - 1:
                print("********* Runing Eval *********")
                eval_loss, eval_pred, eval_label = [], [], []
                for _ in range(optimizer_config['steps_every_eval']):
                    batch_eval_loss, batch_eval_pred, batch_eval_label = sess.run(
                        [eval_tensor_loss, eval_tensor_pred, eval_tensor_label])
                    eval_loss.append(batch_eval_loss)
                    eval_pred.append(batch_eval_pred)
                    eval_label.append(batch_eval_label)

                eval_pred = np.concatenate(eval_pred, axis=0)
                eval_label = np.concatenate(eval_label, axis=0)

                eval_auc = accuracy_score(eval_label.astype(int), (eval_pred > 0.5).astype(int))
                eval_recall = recall_score(eval_label.astype(int), (eval_pred > 0.5).astype(int))
                print("eval_loss: {}".format(np.mean(eval_loss)))
                print("eval_accuracy: {}".format(eval_auc))
                print("eval_recall: {}".format(eval_recall))
                print("********* End Eval *********")

                if (i + 1) % optimizer_config['save_model_steps'] == 0 or i == optimizer_config['num_train_steps'] - 1:
                    # 只有aAuc或召回率提升时，才进行模型的保存
                    if (eval_auc + eval_recall) / 2 > best_score:
                        saver.save(sess, optimizer_config['model_path'], global_step=i, write_meta_graph=False)
                        best_score = (eval_auc + eval_recall) / 2


if __name__ == '__main__':
    tf.logging.set_verbosity("INFO")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_dir', default='config.yaml', type=str)
    parser.add_argument('--action_csv', default='config.yaml', type=str)
    parser.add_argument('--feed_csv', default='feed_emb', type=str)
    parser.add_argument('--maps_file', default='feed_emb', type=str)
    parser.add_argument('--hashtable_ckpt', default=None, type=str)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--config_file', default=None, type=str)
    parser.add_argument('--full_run', default='false', type=str)

    args = parser.parse_args()

    config = yaml.load(open(args.config_file, 'r'))

    run(tfrecord_dir=args.tfrecord_dir,
        action_csv=args.action_csv,
        feed_csv=args.feed_csv,
        map_file=args.maps_file,
        model_config=config['ModelConfig'],
        optimizer_config=config['OptimizerConfig'],
        hashtable_ckpt=args.hashtable_ckpt,
        init_checkpoint=args.init_checkpoint,
        full_run=args.full_run in ['True', 'true'])
