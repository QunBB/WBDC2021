import gc
import multiprocessing
import os
import pickle
import random
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config.config import *
from src.prepare.utils import DataInstance, get_action_df, multiprocess_write


"""
需要内存过多，废弃使用
"""


def run(csv_file, feed_csv_file, maps_file, dataset_dir, pool_num, do_eval=True, k_fold=None,
        mask_rate=None):
    dataset_path = "data_instance"

    if os.path.exists(os.path.join(dataset_dir, dataset_path)) and len(
            os.listdir(os.path.join(dataset_dir, dataset_path))) > 0:
        dataset = []
        for name in os.listdir(os.path.join(dataset_dir, dataset_path)):
            print("Load {}.....".format(name))
            with open(os.path.join(dataset_dir, dataset_path, name), "rb") as file:
                dataset.extend(pickle.load(file))

        print("Load data_instances.pkl end......")

        # dataset_path = "data_instance.pkl"
        # if os.path.exists(os.path.join(dataset_dir, dataset_path)):
        #     with open(os.path.join(dataset_dir, dataset_path), "rb") as file:
        #         dataset = pickle.load(file)
        #
        #     print("Load data_instances.pkl end......")

        print("The number of dataset is: {}".format(len(dataset)))

    else:
        path = os.path.join(dataset_dir, "data_and_count.pkl")
        if os.path.exists(path):
            with open(path, "rb") as file:
                data = pickle.load(file)
                user_count_features_dict = pickle.load(file)
                feed_count_features_dict = pickle.load(file)
                author_count_features_dict = pickle.load(file)
                feed2author = pickle.load(file)

            print("Load data_and_count.pkl ......")
            print("The number of user action is: {}".format(len(data)))

        else:
            data, user_count_features_dict, feed_count_features_dict, author_count_features_dict, feed2author = get_action_df(
                csv_file,
                feed_csv_file,
                maps_file)
            with open(path, "wb") as file:
                pickle.dump(data, file, protocol=4)
                pickle.dump(user_count_features_dict, file)
                pickle.dump(feed_count_features_dict, file)
                pickle.dump(author_count_features_dict, file)
                pickle.dump(feed2author, file)

        dataset = []

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

        columns = list(data.columns)
        userid_col_id = None
        for i, name in enumerate(columns):
            if name == 'userid':
                userid_col_id = i
                break
        assert userid_col_id is not None, columns

        data = data.values
        data_batch = []
        linspace = np.linspace(0, len(all_userid), pool_num * 10 + 1, dtype=int)
        user_num = user_num['feedid'].values
        start = 0
        for i in range(pool_num * 10):
            num = user_num[linspace[i]:linspace[i + 1]].sum()
            data_batch.append((data[start:start + num],
                               user_count_features_dict,
                               feed_count_features_dict,
                               author_count_features_dict,
                               feed2author,
                               columns,
                               user_num[linspace[i]:linspace[i + 1]]))

            start += num
        assert start == len(data)
        assert sum([len(set(d[0][:, userid_col_id])) for d in data_batch]) == len(all_userid), "data_batch: {}".format(
            sum([len(set(d[0][:, userid_col_id])) for d in data_batch]))

        del data
        del user_num
        gc.collect()

        pd.set_option('display.max_columns', None)
        print(pd.DataFrame(data_batch[0][0], columns=data_batch[0][-2]).head())

        # data_batch = [(data.loc[data['userid'] == uid],
        #                user_count_features_dict,
        #                feed_count_features_dict,
        #                author_count_features_dict,
        #                feed2author) for uid in all_userid]

        print("start pool run......")

        # pool = Pool(pool_num)
        # result = pool.map(gene_data_instance, data_batch)
        # pool.close()  # 关闭进程池
        # pool.join()  # 阻塞住进程，等待子进程的运行完毕
        #
        # dataset.extend(list(chain(*result)))
        #
        # print("The number of dataset is: {}".format(len(dataset)))
        #
        # with open(os.path.join(dataset_dir, dataset_path), "wb") as file:
        #     pickle.dump(dataset, file, protocol=4)

        os.system("mkdir -p {}".format(os.path.join(dataset_dir, dataset_path)))
        split_num = 10
        linspace = np.linspace(0, len(data_batch), split_num + 1, dtype=int)
        for i in range(split_num):
            pool = Pool(pool_num)
            # dataset.extend(
            #     chain(*pool.map(gene_data_instance, data_batch[linspace[i]:linspace[i + 1]]))
            # )
            result = pool.map(gene_data_instance, data_batch[linspace[i]:linspace[i + 1]])
            pool.close()  # 关闭进程池
            pool.join()  # 阻塞住进程，等待子进程的运行完毕

            with open(os.path.join(dataset_dir, dataset_path, "instance_{}.pkl".format(i)), "wb") as file:
                pickle.dump(result, file, protocol=4)

            del result
            gc.collect()

        del data_batch
        del user_count_features_dict
        del feed_count_features_dict
        del author_count_features_dict
        del feed2author
        gc.collect()

        for i in range(split_num):
            with open(os.path.join(dataset_dir, dataset_path, "instance_{}.pkl".format(i)), "rb") as file:
                result = pickle.load(file)
            dataset.extend(chain(*result))

            del result
            gc.collect()

        print("The number of all dataset is: {}".format(len(dataset)))

    print("dataset is: ")
    for d in dataset[:10]:
        print(d)

    if do_eval:
        # 生成验证集
        multiprocess_write(dataset=[d for d in dataset if d.date == 14],
                           pool_num=pool_num,
                           dataset_dir=os.path.join(dataset_dir, "tfrecord/eval"))
        dataset = [d for d in dataset if d.date < 14]
    else:
        dataset_dir = os.path.join(dataset_dir, "all_in")

    if mask_rate is not None:
        def _mask(x):
            x.feedid = 0
            return x

        mask_data = random.sample(dataset, int(len(dataset) * mask_rate))
        mask_data = list(map(lambda x: _mask(x), mask_data))

        dataset.extend(mask_data)

    print("The number of dataset is: {}".format(len(dataset)))

    random.shuffle(dataset)

    if k_fold is None:
        tfrecord_dir = ["no_kfold/1"]
        all_dataset = [dataset]
    else:
        tfrecord_dir = ["kfold/" + str(i + 1) for i in range(k_fold)]
        all_dataset = []
        positive_data = list(
            filter(lambda
                       x: x.read_comment == 1 or x.like == 1 or x.click_avatar == 1 or x.forward == 1 or x.comment == 1 or x.follow == 1 or x.favorite == 1,
                   dataset))
        negative_data = list(
            filter(lambda
                       x: x.read_comment == 0 and x.like == 0 and x.click_avatar == 0 and x.forward == 0 and x.comment == 0 and x.follow == 0 and x.favorite == 0,
                   dataset))
        print(
            "++++++++++++++++++++ The Number of Positive Dataset is {} ++++++++++++++++++++".format(len(positive_data)))
        print(
            "++++++++++++++++++++ The Number of Negative Dataset is {} ++++++++++++++++++++".format(len(negative_data)))
        linspace = np.linspace(0, len(negative_data), k_fold + 1)
        for i in range(k_fold):
            one_fold = positive_data + negative_data[linspace[i]:linspace[i + 1]]
            random.shuffle(one_fold)
            all_dataset.append(one_fold)

    del dataset
    gc.collect()

    for i, dataset in enumerate(all_dataset):
        multiprocess_write(dataset=dataset,
                           pool_num=pool_num,
                           dataset_dir=os.path.join(dataset_dir, "tfrecord", tfrecord_dir[i]))
    print("The number of 1-fold dataset is: {}".format(len(dataset)))


def gene_data_instance(paras):
    # data, user_count_features_dict, feed_count_features_dict, author_count_features_dict, feed2author = paras
    data_arr, user_count_features_dict, feed_count_features_dict, author_count_features_dict, feed2author, columns, user_num = paras
    dataset = []

    start = 0
    for num in user_num:
        data = pd.DataFrame(data_arr[start:start + num], columns=columns)
        assert len(data['userid'].drop_duplicates()) == 1

        start += num

        # 一个用户一天 为一个索引
        data.index = data['userid'].astype(str) + "-" + data['date_'].astype(str)

        print("now is the userid: {}".format(data.userid.values[0]))

        for index in set(data.index):
            userid, date = index.split("-")
            if date == '1':
                continue

            count_features_list = []

            # 加入用户的统计特征
            key = "{}|{}".format(userid, date)
            if key not in user_count_features_dict:
                key = "{}|{}".format("unk", date)

            count_features_list.extend(user_count_features_dict[key])

            his = None
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
                    his = data.loc[data.index == "-".join([userid, str(int(date) - i)])]
                    # 播放超过一半，或者 停留时间超过1分钟
                    his = his.loc[(his['play_rate'] >= 0.5) | (his['stay'] >= 1 * 60) | (his['score'] >= 1)]

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

            # TODO 过去n天都没有记录的话，直接忽略，因为赛题说明：测试集都时有历史记录的
            if his is None or len(his_feed) == 0:
                continue

            ctr = data.loc[data.index == index]

            # TODO 这里如果加入float类型的字段，取values就会导致所有字段都变为float
            for feedid_map, device, read_comment, comment, like, click_avatar, \
                forward, follow, favorite, feedid_origin, uid, userid_map in \
                    ctr[
                        ['feedid_map', 'device', 'read_comment', 'comment', 'like', 'click_avatar',
                         'forward', 'follow', 'favorite', 'feedid', 'userid', 'userid_map']].values:

                feed_count_list = []
                # 加入feed的统计特征
                key = "{}|{}".format(feedid_origin, date)
                if key not in feed_count_features_dict:
                    key = "{}|{}".format("unk", date)
                feed_count_list.extend(feed_count_features_dict[key])

                # 加入author的统计特征
                key = "{}|{}".format(feed2author[feedid_origin], date)
                if key not in author_count_features_dict:
                    key = "{}|{}".format("unk", date)
                feed_count_list.extend(author_count_features_dict[key])

                dataset.append(
                    DataInstance(date=int(date),
                                 device=device,
                                 read_comment=read_comment,
                                 comment=comment,
                                 like=like,
                                 click_avatar=click_avatar,
                                 forward=forward,
                                 follow=follow,
                                 favorite=favorite,
                                 feedid_origin_history=his_feed,
                                 behaviors=his_behaviors,
                                 feedid_seq_len=len(his_feed),
                                 feedid_origin=feedid_origin,
                                 userid=userid_map,
                                 userid_origin=uid,
                                 interval_history=his_interval,
                                 count_features=count_features_list + feed_count_list
                                 ))
    return dataset


if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', default='feed_embeddings.csv', type=str)
    parser.add_argument('--feed_csv_file', default='feed_info.csv', type=str)
    parser.add_argument('--maps_file', default='map.pkl', type=str)
    parser.add_argument('--dataset_dir', default='feed_emb', type=str)
    parser.add_argument('--pool_num', default=8, type=int)
    parser.add_argument('--do_eval', default="Y", type=str)
    parser.add_argument('--k_fold', default=None, type=int)
    parser.add_argument('--mask_rate', default=None, type=float)

    args = parser.parse_args()

    run(args.csv_file, args.feed_csv_file, args.maps_file,
        args.dataset_dir, args.pool_num,
        do_eval=args.do_eval in ["Y", "y", "yes", "Yes", "True", "true"],
        k_fold=args.k_fold,
        mask_rate=args.mask_rate)

    """
    根据不同的平台， multiprocessing 支持三种启动进程的方法。这些 启动方法 有

    spawn
    父进程会启动一个全新的 python 解释器进程。 子进程将只继承那些运行进程对象的 run() 方法所必需的资源。 特别地，来自父进程的非必需文件描述符和句柄将不会被继承。 使用此方法启动进程相比使用 fork 或 forkserver 要慢上许多。
    
    可在Unix和Windows上使用。 Windows上的默认设置。
    
    fork
    父进程使用 os.fork() 来产生 Python 解释器分叉。子进程在开始时实际上与父进程相同。父进程的所有资源都由子进程继承。请注意，安全分叉多线程进程是棘手的。
    
    只存在于Unix。Unix中的默认值。
    
    forkserver
    程序启动并选择* forkserver * 启动方法时，将启动服务器进程。从那时起，每当需要一个新进程时，父进程就会连接到服务器并请求它分叉一个新进程。分叉服务器进程是单线程的，因此使用 os.fork() 是安全的。没有不必要的资源被继承。
    
    可在Unix平台上使用，支持通过Unix管道传递文件描述符。
    """
