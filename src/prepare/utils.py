import tensorflow as tf
import collections
import os
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from recordclass import make_dataclass, dataobject

from config.config import *


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))  # 需要注意这里接受的格式是list，并且只能是一维的
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def create_bytes_feature(values):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    return f


def write_example_into_file(paras_list):
    dataset, save_path, index = paras_list
    print("write_example_into_file: ".format(index))
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, str(index) + ".tfrecord"))

    count = 1
    for instance in dataset:
        example = write_example(instance)
        writer.write(example.SerializeToString())

        if count % 1000 == 0:
            print("example index [{}] finish: {}".format(index, count))
        count += 1

    writer.close()


def write_example(instance):
    features = collections.OrderedDict()
    features['device'] = create_int_feature([instance.device])
    features['read_comment'] = create_int_feature([instance.read_comment])
    features['comment'] = create_int_feature([instance.comment])
    features['like'] = create_int_feature([instance.like])
    features['click_avatar'] = create_int_feature([instance.click_avatar])
    features['forward'] = create_int_feature([instance.forward])
    features['follow'] = create_int_feature([instance.follow])
    features['favorite'] = create_int_feature([instance.favorite])
    features['userid'] = create_int_feature([instance.userid])
    features['feedid_origin'] = create_int_feature([instance.feedid_origin])

    features['userid_origin'] = create_int_feature([instance.userid_origin])

    count_features = instance.count_features
    features['count_features'] = create_float_feature(np.array(count_features, dtype=np.float32))

    history = list(instance.feedid_origin_history)
    behaviors = list(instance.behaviors)
    interval_history = list(instance.interval_history)
    if len(history) >= HISTORY_MAX_LEN:
        history = history[:HISTORY_MAX_LEN]
        behaviors = behaviors[:HISTORY_MAX_LEN]
        interval_history = interval_history[:HISTORY_MAX_LEN]
        features['history_seq_len'] = create_int_feature([HISTORY_MAX_LEN])
    else:
        features['history_seq_len'] = create_int_feature([len(history)])
        history.extend([0] * (HISTORY_MAX_LEN - len(history)))
        interval_history.extend([0] * (HISTORY_MAX_LEN - len(interval_history)))
        behaviors.extend([",".join(["0"] * 7)] * (HISTORY_MAX_LEN - len(behaviors)))
    features['feedid_origin_history'] = create_int_feature(history)
    features['interval_history'] = create_int_feature(interval_history)

    # 0.01为平滑项，防止全部为0，权重也为0
    behaviors = np.array(list(map(lambda x: x.split(","), behaviors)),
                         dtype=np.float32)
    behaviors = np.maximum(behaviors, 0.01)
    features['behaviors'] = create_bytes_feature(behaviors.tostring())

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


def multiprocess_write(dataset, pool_num, dataset_dir):
    os.system("mkdir -p {}".format(dataset_dir))
    linspace = np.linspace(0, len(dataset), pool_num + 1, dtype=int)
    train_paras_list = [(dataset[linspace[i]:linspace[i + 1]],
                         dataset_dir,
                         i) for i in range(pool_num)]
    pool = Pool(pool_num)
    _ = pool.map(write_example_into_file, train_paras_list)
    pool.close()  # 关闭进程池
    pool.join()  # 阻塞住进程，等待子进程的运行完毕


def get_action_df(csv_file, feed_csv_file, maps_file):
    data = pd.DataFrame()
    for csv in csv_file.split(","):
        data = data.append(pd.read_csv(csv, dtype=int))

    print("The number of user action is: {}".format(len(data)))

    with open(maps_file, 'rb') as file:
        mappings = pickle.load(file)

    feed_map = mappings['feedid']

    feed_info = pd.read_csv(feed_csv_file)[['feedid', 'authorid', 'videoplayseconds']]
    feed_info['authorid'] = feed_info['authorid'].fillna(-1)

    feed2author = {v[0]: v[1] for v in feed_info[['feedid', 'authorid']].values}

    data = pd.merge(data, feed_info, on='feedid')

    # 毫秒转化为秒
    data['play'] = data['play'] / 1000
    data['stay'] = data['stay'] / 1000

    # 过滤停留时间过短
    # 过滤了会导致部分测试集用户没有历史记录
    data['play_rate'] = data['play'] / data['videoplayseconds']
    data['stay_rate'] = data['stay'] / data['videoplayseconds']

    # data = data.loc[(data.stay >= 1000) & (data.play >= 1000)]

    def dict_count_features(subject_name):
        subject_map = {"userid": "feedid",
                       "feedid": "userid",
                       "authorid": "userid"}

        result = {}
        features_name = []
        # 前一天的统计特征
        count = {label: "sum" for label in LABELS_NAME}
        count.update({subject_map[subject_name]: "count", "play_rate": "mean", "stay_rate": "mean", "play": "mean",
                      "stay": "mean"})
        subject_count_features = data.groupby([subject_name, 'date_'], as_index=False)[list(count.keys())].agg(count)
        features_name.extend(["{}_{}".format(k, v) for k, v in count.items()])
        subject_count_features = subject_count_features.rename(
            columns={k: "{}_{}".format(k, v) for k, v in count.items()})
        # 标准差
        count = {"play_rate": "std", "stay_rate": "std", "play": "std", "stay": "std"}
        subject_std_features = data.groupby([subject_name, 'date_'], as_index=False)[list(count.keys())].agg(
            count).rename(
            columns={k: "{}_{}".format(k, v) for k, v in count.items()})
        for name in ["{}_{}".format(k, v) for k, v in count.items()]:
            subject_std_features.loc[subject_std_features[name].isna(), name] = 0
        subject_count_features = pd.merge(subject_count_features, subject_std_features, on=[subject_name, 'date_'])
        features_name.extend(["{}_{}".format(k, v) for k, v in count.items()])
        subject_count_features['date_'] = subject_count_features['date_'] + 1
        # 前一天未出现的，则置为0
        unk = pd.DataFrame(data=[['unk', i] + [0] * len(features_name) for i in range(2, 16)],
                           columns=[subject_name, 'date_'] + features_name)
        subject_count_features = subject_count_features.append(unk)

        all_features = features_name

        # 前n天的统计特征
        subject_count_features_agg = pd.DataFrame()
        for i in range(2, 16):
            data_tmp = data.loc[data['date_'] < i]
            data_tmp['date_'] = i

            features_name = []

            # 计数的统计特征
            count = {label: "sum" for label in LABELS_NAME}
            count.update({subject_map[subject_name]: "count", "play_rate": "mean", "stay_rate": "mean", "play": "mean",
                          "stay": "mean"})
            subject_count_features_tmp = data_tmp.groupby([subject_name, 'date_'], as_index=False)[
                list(count.keys())].agg(
                count).rename(
                columns={k: "agg_{}_{}".format(k, v) for k, v in count.items()})
            subject_count_features_tmp["agg_{}_{}".format(subject_map[subject_name], "count")] = \
                subject_count_features_tmp["agg_{}_{}".format(subject_map[subject_name], "count")] / (
                        subject_count_features_tmp['date_'] - 1)
            features_name.extend(["agg_{}_{}".format(k, v) for k, v in count.items()])
            subject_count_features_tmp = subject_count_features_tmp.rename(
                columns={k: "agg_{}_{}".format(k, v) for k, v in count.items()})
            # 标准差
            count = {"play_rate": "std", "stay_rate": "std", "play": "std", "stay": "std"}
            subject_std_features_tmp = data_tmp.groupby([subject_name, 'date_'], as_index=False)[
                list(count.keys())].agg(count).rename(
                columns={k: "agg_{}_{}".format(k, v) for k, v in count.items()})
            for name in ["agg_{}_{}".format(k, v) for k, v in count.items()]:
                subject_std_features_tmp.loc[subject_std_features_tmp[name].isna(), name] = 0
            subject_count_features_tmp = pd.merge(subject_count_features_tmp, subject_std_features_tmp,
                                                  on=[subject_name, 'date_'])
            features_name.extend(["agg_{}_{}".format(k, v) for k, v in count.items()])

            subject_count_features_agg = subject_count_features_agg.append(subject_count_features_tmp)

        # 前n天未出现的，置为0
        unk = pd.DataFrame(data=[['unk', i] + [0] * len(features_name) for i in range(2, 16)],
                           columns=[subject_name, 'date_'] + features_name)
        subject_count_features_agg = subject_count_features_agg.append(unk)

        # TODO 应为右连接
        final_count = pd.merge(subject_count_features, subject_count_features_agg,
                               on=[subject_name, 'date_'], how='right')
        # 此时的all_features 为前一天的统计特征name
        for name in all_features:
            final_count[name] = final_count[name].fillna(0)

        pd.set_option('display.max_columns', None)

        print(final_count.head())

        all_features.extend(features_name)

        print(all_features)

        # 对统计特征进行标准化处理
        scaler = StandardScaler().fit(final_count[all_features].values)
        scaler_res = scaler.transform(final_count[all_features].values)
        for i, name in enumerate(all_features):
            final_count[name] = scaler_res[:, i]

        print(final_count.head())

        for value in final_count[[subject_name, 'date_'] + all_features].values:
            key = "{}|{}".format(value[0], value[1])
            result[key] = value[2:].tolist()

        return result

    user_count_features_dict = dict_count_features('userid')
    feed_count_features_dict = dict_count_features('feedid')
    author_count_features_dict = dict_count_features('authorid')

    data['score'] = data['read_comment'] + data['comment'] + data['like'] + data['click_avatar'] + data['forward'] + \
                    data['follow'] + data['favorite']

    data['behaviors'] = (data['read_comment']).astype(str).str.cat((data[
        ['comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']]).astype(str), sep=',')

    # device取值为1、2
    data['device'] = data['device'] - 1

    data['feedid_map'] = data['feedid'].apply(lambda x: feed_map.get(x) or 0)

    data['userid_map'] = data['userid'].apply(lambda x: mappings['userid'][x])

    # 一个用户一天 为一个索引
    # data.index = data['userid'].astype(str) + "-" + data['date_'].astype(str)

    # 计算feed第一次出现的date，用于模拟冷启动feed
    feed_first_day = data.groupby('feedid', as_index=False)['date_'].min()
    feed_first_day = feed_first_day.rename(columns={"date_": "first_date"})
    data = pd.merge(data, feed_first_day, on='feedid')

    print("utils.py", data.dtypes)

    return data, user_count_features_dict, feed_count_features_dict, author_count_features_dict, feed2author


class DataInstance(dataobject):
    date: int
    first_date: int
    device: int
    read_comment: int
    comment: int
    like: int
    click_avatar: int
    forward: int
    follow: int
    favorite: int
    feedid_origin_history: list
    behaviors: list
    feedid_seq_len: int
    feedid_origin: int
    userid: int
    userid_origin: int
    interval_history: list
    count_features: list

    # def __init__(self, date, device, read_comment, comment, like, click_avatar, forward, follow,
    #              favorite, feedid_origin_history, behaviors, feedid_seq_len, feedid_origin, userid, userid_origin,
    #              interval_history, count_features):
    #     self.date = date
    #     self.device = device
    #     self.read_comment = read_comment
    #     self.comment = comment
    #     self.like = like
    #     self.click_avatar = click_avatar
    #     self.forward = forward
    #     self.follow = follow
    #     self.favorite = favorite
    #     self.feedid_origin_history = feedid_origin_history
    #     self.behaviors = behaviors
    #     self.feedid_seq_len = feedid_seq_len
    #     self.feedid_origin = feedid_origin
    #     self.userid = userid
    #     self.userid_origin = userid_origin
    #     self.interval_history = interval_history
    #     self.count_features = count_features

    # instance = make_dataclass("DataInstance",
    #                           [("date", int),
    #                            ("device", int),
    #                            ("read_comment", int),
    #                            ("comment", int),
    #                            ("like", int),
    #                            ("click_avatar", int),
    #                            ("forward", int),
    #                            ("follow", int),
    #                            ("favorite", int),
    #                            ("feedid_origin_history", list),
    #                            ("behaviors", list),
    #                            ("feedid_seq_len", int),
    #                            ("feedid_origin", int),
    #                            ("userid", int),
    #                            ("userid_origin", int),
    #                            ("interval_history", list),
    #                            ("count_features", list)
    #                            ])
    #
    # def __call__(self, date, device, read_comment, comment, like, click_avatar, forward, follow,
    #              favorite, feedid_origin_history, behaviors, feedid_seq_len, feedid_origin, userid, userid_origin,
    #              interval_history, count_features):
    #     return self.instance(date=date,
    #                          device=device,
    #                          read_comment=read_comment,
    #                          comment=comment,
    #                          like=like,
    #                          click_avatar=click_avatar,
    #                          forward=forward,
    #                          follow=follow,
    #                          favorite=favorite,
    #                          feedid_origin_history=feedid_origin_history,
    #                          behaviors=behaviors,
    #                          feedid_seq_len=feedid_seq_len,
    #                          feedid_origin=feedid_origin,
    #                          userid=userid,
    #                          userid_origin=userid_origin,
    #                          interval_history=interval_history,
    #                          count_features=count_features)
