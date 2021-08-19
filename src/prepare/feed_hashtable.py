import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import os

from config.config import *


def run(feed_csv_file, map_file, save_path):
    with open(map_file, "rb") as file:
        mappings = pickle.load(file)

    def _videoplayseconds_max_sparse(x):
        for i, n in enumerate([5, 10, 20, 30, 40, 50, 60]):
            if x < n:
                return i
        return i + 1

    feed_info = pd.read_csv(feed_csv_file)

    feed_info['feedid_origin'] = feed_info['feedid']

    # 将videoplayseconds离散化
    feed_info['videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x: _videoplayseconds_max_sparse(x))

    # 缺失值补-1
    for name in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
        feed_info[name] = feed_info[name].fillna(-1)
    feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna("-1")
    feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].fillna("-1")
    feed_info['description'] = feed_info['description'].fillna("-1")

    # 将原值映射为id
    for name in ['authorid', 'bgm_song_id', 'bgm_singer_id', 'feedid']:
        feed_info[name] = feed_info[name].apply(lambda x: mappings[name][x])
    feed_info['manual_tag_list'] = feed_info['manual_tag_list'].apply(
        lambda x: ";".join([str(mappings['tag'][v]) for v in x.split(";")]))
    feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].apply(
        lambda x: ";".join([str(mappings['keyword'][v]) for v in x.split(";")]))
    feed_info['description'] = feed_info['description'].apply(
        lambda x: ";".join([str(mappings['word'][v]) for v in str(x).split(" ")]))

    keys_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    values_tensor = tf.placeholder(dtype=tf.int32, shape=[None])
    insert_table = {}
    hashtable = {}
    keys_input = feed_info['feedid_origin'].values
    values_input = {}
    for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds']:
        hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                             name="HashTable_" + name)
        insert_table[name] = hashtable[name].insert(keys_tensor, values_tensor)
        values_input[name] = feed_info[name].values

    # 原始的feedid映射到 0-max(feedid)
    name = "feedid_origin_id"
    hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                         name="HashTable_" + name)
    insert_table[name] = hashtable[name].insert(keys_tensor, values_tensor)
    values_input[name] = list(range(len(feed_info)))

    for name, tag in zip(['manual_tag_list', 'description', 'manual_keyword_list'], ['tag', 'word', 'keyword']):
        # hashtable[tag] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
        #                                                     name="HashTable_" + tag)
        # insert_table[tag] = hashtable[tag].insert(keys_tensor, values_tensor)
        hashtable[tag + "_len"] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32,
                                                                     default_value=0,
                                                                     name="HashTable_" + tag + "_len")
        insert_table[tag + "_len"] = hashtable[tag + "_len"].insert(keys_tensor, values_tensor)

        ids_list = []
        len_list = []
        for value in feed_info[name].values:
            tmp = value.split(";")
            tmp = list(map(lambda x: int(x), tmp))
            if len(tmp) >= CTR_MAX_LEN[tag]:
                ids_list.append(tmp[:CTR_MAX_LEN[tag]])
                len_list.append(CTR_MAX_LEN[tag])
            else:
                len_list.append(len(tmp))
                tmp.extend([0] * (CTR_MAX_LEN[tag] - len(tmp)))
                ids_list.append(tmp)
        values_input[tag] = ids_list
        values_input[tag + "_len"] = len_list

    # MutableHashTable无法映射数组，只能通过embedding_lookup的方式进行映射
    with tf.variable_scope("hashtable"):
        for name in ['tag', 'word', 'keyword']:
            table = tf.get_variable(name + "_hashtable",
                                    initializer=np.array(values_input[name], dtype=np.int32),
                                    dtype=tf.int32,
                                    trainable=False)
            print("The table of {} is {}".format(name, table.shape))

    os.system("mkdir -p {}".format(os.path.dirname(save_path)))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds', 'feedid_origin_id']:
            sess.run(insert_table[name], feed_dict={keys_tensor: keys_input,
                                                    values_tensor: values_input[name]})

        for name in ['tag', 'word', 'keyword']:
            # sess.run(insert_table[name], feed_dict={keys_tensor: keys_input,
            #                                         values_tensor: values_input[name]})
            sess.run(insert_table[name + "_len"], feed_dict={keys_tensor: keys_input,
                                                             values_tensor: values_input[name + "_len"]})

        saver.save(sess, save_path)

    print(tf.train.list_variables(save_path))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feed_csv', default='feed_info.csv', type=str)
    parser.add_argument('--map_file', default='test_a.csv', type=str)
    parser.add_argument('--save_path', default='map.pkl', type=str)

    args = parser.parse_args()

    run(args.feed_csv, args.map_file, args.save_path)
