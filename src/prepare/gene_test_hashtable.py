import multiprocessing
import pickle
from multiprocessing import Pool
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gc

from config.config import *


def run(feed_csv_file, features_file, map_file, pool_num, train_sess_path, save_path):
    with open(features_file, "rb") as file:
        data = pickle.load(file)
        user_count_features_dict = pickle.load(file)
        feed_count_features_dict = pickle.load(file)
        author_count_features_dict = pickle.load(file)
        # feed2author = pickle.load(file)

    print("load features_file")

    print(
        "非冷启动user数量为: {}".format(len([name for name in user_count_features_dict.keys() if name.split("|")[1] == "15"])))
    print(
        "非冷启动feed数量为: {}".format(len([name for name in feed_count_features_dict.keys() if name.split("|")[1] == "15"])))
    print("非冷启动author数量为: {}".format(
        len([name for name in author_count_features_dict.keys() if name.split("|")[1] == "15"])))

    with open(map_file, 'rb') as file:
        mappings = pickle.load(file)

    feed_info = pd.read_csv(feed_csv_file)
    feed_info['authorid'] = feed_info['authorid'].fillna(-1)
    feedid_list = []
    feed_result = []
    for feedid, authorid in feed_info[['feedid', 'authorid']].values:
        feedid_list.append(feedid)
        feed_values = feed_count_features_dict.get("{}|{}".format(feedid, 15)) or feed_count_features_dict["unk|15"]
        author_values = author_count_features_dict.get("{}|{}".format(authorid, 15)) or author_count_features_dict[
            "unk|15"]
        assert isinstance(feed_values, list) and isinstance(author_values, list)
        feed_result.append(feed_values + author_values)

    # 为了防止feed_info中的feedid顺序不同，导致读取feed特征错误
    name = "feedid_origin_id"
    feed_hashtable = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                        name="HashTable_" + name)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, train_sess_path)

        feed_order = sess.run(feed_hashtable.lookup(feedid_list))

    assert min(feed_order) == 0 and max(feed_order) == len(feed_order) - 1
    assert len(feed_order) == len(feedid_list) == len(feed_result)
    feed_tuple = [(feed_order[i], feedid_list[i], feed_result[i]) for i in range(len(feed_order))]
    feed_tuple = sorted(feed_tuple, key=lambda x: x[0])
    for i in range(5):
        print(feed_tuple[i][0], feed_tuple[i][1], feed_tuple[i][2])

    tf.reset_default_graph()

    for name in ['userid', 'feedid']:
        data[name] = data[name].astype(int)

    # 一个用户一个进程来处理
    all_userid = data['userid'].drop_duplicates().values

    data.sort_values(by="userid", inplace=True)
    user_num = data.groupby('userid', as_index=False)['feedid'].count()
    user_num.sort_values(by="userid", inplace=True)
    assert len(data) == user_num['feedid'].sum()
    assert len(user_num) == len(all_userid)
    assert len(set.difference(set(user_num['userid'].values), set(all_userid))) == 0
    assert data['userid'].values[0] == user_num['userid'].values[0] and data['userid'].values[-1] == \
           user_num['userid'].values[-1]

    print("start generate paras.....")
    start = 0
    columns = list(data.columns)
    data = data.values
    data_batch = []
    for num in tqdm(user_num['feedid'].values, total=len(all_userid)):
        data_batch.append((data[start:start + num],
                           user_count_features_dict,
                           columns))
        start += num
    assert start == len(data)

    del data
    del user_num
    gc.collect()

    pd.set_option('display.max_columns', None)
    print(pd.DataFrame(data_batch[0][0], columns=data_batch[0][-1]).head())

    # data_batch = [(data.loc[data['userid'] == uid],
    #                user_count_features_dict) for uid in all_userid]

    pool = Pool(pool_num)

    user_result = pool.map(gene_data_instance, data_batch)
    pool.close()  # 关闭进程池
    pool.join()  # 阻塞住进程，等待子进程的运行完毕

    for r in user_result[:10]:
        print(r)
    print("The number of user is {}".format(len(user_result)))

    hashtable = {}

    inputs_dict = {name: [] for name in
                   ['userid_origin_id', 'userid_origin', 'userid', 'feedid_origin_history', 'his_behaviors',
                    'interval_history', 'user_count_features', 'history_seq_len']}

    for number, ret in enumerate(user_result):
        for i, name in enumerate(
                ['userid_origin', 'feedid_origin_history', 'his_behaviors', 'interval_history', 'user_count_features',
                 'history_seq_len']):
            inputs_dict[name].append(ret[i])
            if name == 'userid_origin':
                inputs_dict['userid'].append(mappings['userid'][ret[i]])
        inputs_dict['userid_origin_id'].append(number)

    for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds', 'feedid_origin_id',
                 'userid_origin_id', 'userid', 'history_seq_len']:
        hashtable[name] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
                                                             name="HashTable_" + name)

    for tag in ['tag', 'word', 'keyword']:
        # hashtable[tag] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32, default_value=0,
        #                                                     name="HashTable_" + tag)
        # insert_table[tag] = hashtable[tag].insert(keys_tensor, values_tensor)
        hashtable[tag + "_len"] = tf.contrib.lookup.MutableHashTable(key_dtype=tf.int32, value_dtype=tf.int32,
                                                                     default_value=0,
                                                                     name="HashTable_" + tag + "_len")

    inputs_dict['feed_count_features'] = [t[2] for t in feed_tuple]
    print("feed_count_features:")
    print(inputs_dict['feed_count_features'][:5])

    table = {}
    # MutableHashTable无法映射数组，只能通过embedding_lookup的方式进行映射
    with tf.variable_scope("hashtable"):
        for name in ['tag', 'word', 'keyword']:
            table[name] = tf.get_variable(name + "_hashtable",
                                          shape=[len(feed_info), CTR_MAX_LEN[name]],
                                          dtype=tf.int32,
                                          trainable=False)

        for name in ['feedid_origin_history', 'interval_history']:
            table[name] = tf.get_variable(name + "_hashtable",
                                          initializer=np.array(inputs_dict[name], dtype=np.int32),
                                          dtype=tf.int32,
                                          trainable=False)
            print("The table of {} is {}".format(name, table[name].shape))

        for name in ['his_behaviors', 'user_count_features', 'feed_count_features']:
            table[name] = tf.get_variable(name + "_hashtable",
                                          initializer=np.array(inputs_dict[name], dtype=np.float32),
                                          dtype=tf.float32,
                                          trainable=False)
            print("The table of {} is {}".format(name, table[name].shape))

    train_saver = tf.train.Saver(var_list=[hashtable[name] for name in
                                           ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds',
                                            'feedid_origin_id']] +
                                          [table[name] for name in ['tag', 'word', 'keyword']] +
                                          [hashtable[name + "_len"] for name in ['tag', 'word', 'keyword']])
    all_saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_saver.restore(sess, train_sess_path)

        sess.run(hashtable['userid_origin_id'].insert(inputs_dict['userid_origin'], inputs_dict['userid_origin_id']))
        sess.run(hashtable['userid'].insert(inputs_dict['userid_origin'], inputs_dict['userid']))
        sess.run(hashtable['history_seq_len'].insert(inputs_dict['userid_origin'], inputs_dict['history_seq_len']))

        all_saver.save(sess, save_path)

    print(tf.train.list_variables(save_path))


def gene_data_instance(paras):
    data, user_count_features_dict, columns = paras
    data = pd.DataFrame(data, columns=columns)
    assert len(data['userid'].drop_duplicates()) == 1

    # 一个用户一天 为一个索引
    data.index = data['userid'].astype(str) + "-" + data['date_'].astype(str)

    userid = data['userid'].values[0]
    date = 15

    print("userid: {}".format(userid))

    # 加入用户的统计特征
    key = "{}|{}".format(userid, date)
    if key not in user_count_features_dict:
        key = "unk|15"

    his_feed = []
    his_interval = []
    his_behaviors = []

    for i in range(1, int(date) + 1):

        if i == int(date):
            # 没有高质量点击序列，则只保留play最大的一条
            if len(his_feed) > 0:
                break
            his = data.loc[data['date_'] < i]
            his = his.sort_values(by='play').tail(1)
        else:  # 优先取质量高的历史点击序列
            his = data.loc[data.index == "-".join([str(userid), str(int(date) - i)])]
            # 播放超过一半，或者 停留时间超过1分钟
            his = his.loc[(his['play_rate'] >= 0.5) | (his['stay'] >= 1 * 60 * 1000) | (his['score'] >= 1)]

        if len(his) == 0:
            continue

        # 将his进行翻转，让最近的记录在前面
        his['order'] = list(range(len(his)))
        his = his.sort_values(by='order', ascending=False)

        if len(his_feed) + len(his) >= HISTORY_MAX_LEN:
            his_feed.extend(his['feedid'].tail(HISTORY_MAX_LEN - len(his_feed)).values.tolist())
            his_behaviors.extend(his['behaviors'].tail(HISTORY_MAX_LEN - len(his_behaviors)).values.tolist())
            his_interval.extend([i - 1] * (HISTORY_MAX_LEN - len(his_interval)))
            break
        else:
            his_feed.extend(his['feedid'].values.tolist())
            his_behaviors.extend(his['behaviors'].values.tolist())
            his_interval.extend([i - 1] * len(his))

    assert len(his_feed) > 0 and len(
        his_feed) <= HISTORY_MAX_LEN, "The user history length is {}, HISTORY_MAX_LEN is {}".format(
        len(his_feed), HISTORY_MAX_LEN)

    his_len = min(len(his_feed), HISTORY_MAX_LEN)

    if len(his_feed) < HISTORY_MAX_LEN:
        his_feed.extend([0] * (HISTORY_MAX_LEN - len(his_feed)))
        his_behaviors.extend([",".join(["0"] * 7)] * (HISTORY_MAX_LEN - len(his_behaviors)))
        his_interval.extend([0] * (HISTORY_MAX_LEN - len(his_interval)))

    # 0.01为平滑项，防止全部为0，权重也为0
    his_behaviors = np.array(list(map(lambda x: x.split(","), his_behaviors)),
                             dtype=np.float32)
    his_behaviors = np.maximum(his_behaviors, 0.01)

    return userid, his_feed, his_behaviors, his_interval, user_count_features_dict[key], his_len


if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feed_csv_file', default='feed_embeddings.csv', type=str)
    parser.add_argument('--features_file', default='feed_info.csv', type=str)
    parser.add_argument('--maps_file', default='feed_info.csv', type=str)
    parser.add_argument('--pool_num', default=8, type=int)
    parser.add_argument('--train_sess_path', default='feed_emb', type=str)
    parser.add_argument('--save_path', default='feed_emb', type=str)

    args = parser.parse_args()

    run(args.feed_csv_file, args.features_file, args.maps_file, args.pool_num,
        args.train_sess_path, args.save_path)
