from collections import defaultdict

import numpy as np
from numba import njit
from scipy.stats import rankdata

import tensorflow as tf

from config.config import *


def get_tf_hashtable(feed_num, user_num=None):
    hashtable = {}

    for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds']:
        hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                             name="HashTable_" + name,
                                                             )

    # 原始的feedid映射到 0-max(feedid)
    name = "feedid_origin_id"
    hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                         name="HashTable_" + name,
                                                         )

    for tag in ['tag', 'word', 'keyword']:
        hashtable[tag + "_len"] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32,
                                                                     default_value=0,
                                                                     name="HashTable_" + tag + "_len",
                                                                     )

    table = {}
    # MutableHashTable无法映射数组，只能通过embedding_lookup的方式进行映射
    with tf.variable_scope("hashtable"):
        for name in ['tag', 'word', 'keyword']:
            table[name] = tf.get_variable(name + "_hashtable",
                                          shape=[feed_num, CTR_MAX_LEN[name]],
                                          dtype=tf.int32,
                                          trainable=False)
            print("The table of {} is {}".format(name, table[name].shape))

    if user_num is not None:
        for name in ['userid_origin_id', 'userid', 'history_seq_len']:
            hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32,
                                                                 default_value=0,
                                                                 name="HashTable_" + name,
                                                                 )
        with tf.variable_scope("hashtable"):
            table['feedid_origin_history'] = tf.get_variable("feedid_origin_history_hashtable",
                                                             shape=[user_num, HISTORY_MAX_LEN],
                                                             dtype=tf.int32,
                                                             trainable=False)
            table['interval_history'] = tf.get_variable("interval_history_hashtable",
                                                        shape=[user_num, HISTORY_MAX_LEN],
                                                        dtype=tf.int32,
                                                        trainable=False)
            table['his_behaviors'] = tf.get_variable("his_behaviors_hashtable",
                                                     shape=[user_num, HISTORY_MAX_LEN, len(LABELS_NAME)],
                                                     dtype=tf.float32,
                                                     trainable=False)
            table['user_count_features'] = tf.get_variable("user_count_features_hashtable",
                                                           shape=[user_num, COUNT_FEATURES_LEN / 3],
                                                           dtype=tf.float32,
                                                           trainable=False)
            table['feed_count_features'] = tf.get_variable("feed_count_features_hashtable",
                                                           shape=[feed_num, COUNT_FEATURES_LEN / 3 * 2],
                                                           dtype=tf.float32,
                                                           trainable=False)

    return list(hashtable.values()) + list(table.values()), hashtable, table


@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = fast_auc(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc) / size
    return user_auc


def compute_weighted_score(predction_df, weights_map):
    '''评测结果: 多个行为的加权uAUC分数
    Input:
        result_data: 提交的结果文件，二进制格式
        label_data: 对应的label文件，二进制格式
        mode: 比赛阶段，String. "初赛"/"复赛"
    Output:
        result: 评测结果，dict
    '''
    actions = list(weights_map.keys())
    result_actions = []
    label_actions = []
    for action in actions:
        result_actions.append("result_" + action)
        label_actions.append("label_" + action)

    # 计算分数
    y_true = predction_df[label_actions].astype(int).values
    y_pred = predction_df[result_actions].astype(float).values.round(decimals=6)
    userid_list = predction_df['userid'].astype(str).tolist()
    del predction_df
    score = 0.0
    weights_sum = 0.0
    score_detail = {}

    for i, action in enumerate(actions):
        # print(action)
        y_true_bev = y_true[:, i]
        y_pred_bev = y_pred[:, i]
        weight = weights_map[action]
        # user AUC
        uauc = uAUC(y_true_bev, y_pred_bev, userid_list)
        # print(uauc)
        score_detail[action] = round(uauc, 6)
        score += weight * uauc
        weights_sum += weight
    score /= weights_sum
    score = round(score, 6)

    score_detail['weighted'] = score

    return score_detail
