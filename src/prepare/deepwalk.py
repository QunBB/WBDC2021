import os
import pickle
import random
from multiprocessing import Pool
import multiprocessing

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from config.config import RANDOM_SEED

random.seed(RANDOM_SEED)


def run(action_csv, test_csv, feed_csv, save_dir, word2vec_conf, deepwalk_conf, map_file):
    with open(map_file, "rb") as file:
        mapping = pickle.load(file)

    sentences = None
    nodes = None
    node_sides = None

    if os.path.exists(os.path.join(save_dir, "walk_path.pkl")):
        with open(os.path.join(save_dir, "walk_path.pkl"), "rb") as file:
            sentences = pickle.load(file)

    if sentences is None:
        if os.path.exists(os.path.join(save_dir, "graph.obj")):
            with open(os.path.join(save_dir, "graph.obj"), "rb") as file:
                nodes = pickle.load(file)
                node_sides = pickle.load(file)

        action_df = pd.DataFrame()
        for csv in action_csv.split(","):
            action_df = action_df.append(pd.read_csv(csv))

        # test_df = pd.DataFrame()
        # for name in test_csv.split(","):
        #     test_df = test_df.append(pd.read_csv(name)[['userid', 'feedid']])

        feed_df = pd.read_csv(feed_csv)[[
            'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list', 'manual_keyword_list', 'videoplayseconds']]

        for name in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
            feed_df[name] = feed_df[name].fillna(-1)
        feed_df['manual_tag_list'] = feed_df['manual_tag_list'].fillna("-1")
        feed_df['manual_keyword_list'] = feed_df['manual_keyword_list'].fillna("-1")

        for name in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
            feed_df[name] = feed_df[name].apply(lambda x: mapping[name][x])
        feed_df['manual_tag_list'] = feed_df['manual_tag_list'].apply(
            lambda x: ";".join([str(mapping['tag'][v]) for v in x.split(";")]))
        feed_df['manual_keyword_list'] = feed_df['manual_keyword_list'].apply(
            lambda x: ";".join([str(mapping['keyword'][v]) for v in x.split(";")]))

        data = action_df
        # data = action_df.append(test_df)
        data = pd.merge(data, feed_df, on='feedid')
        data['userid'] = data['userid'].apply(lambda x: mapping['userid'][x])
        data['feedid'] = data['feedid'].apply(lambda x: mapping['feedid'][x])
        data['tag'] = data['manual_tag_list']
        data['keyword'] = data['manual_keyword_list']

        data['score'] = data['read_comment'] + data['comment'] + data['like'] + data['click_avatar'] + data['forward'] + \
                        data['follow'] + data['favorite']
        data['play_rate'] = data['play'] / 1000 / data['videoplayseconds']
        data = data.loc[(data['play_rate'] >= 0.5) | (data['stay'] >= 60 * 1000) | (data['score'] >= 1)]

    if sentences is None and nodes is None:
        nodes = set()
        node_sides = {}

        for arr in tqdm(data[['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list',
                              'manual_keyword_list']].values,
                        total=len(data)):

            userid, feedid, authorid, bgm_song_id, bgm_singer_id, manual_tag_list, manual_keyword_list = arr
            userid = "userid-" + str(userid)
            feedid = "feedid-" + str(feedid)
            authorid = "authorid-" + str(authorid)
            bgm_song_id = "bgm_song_id-" + str(bgm_song_id)
            bgm_singer_id = "bgm_singer_id-" + str(bgm_singer_id)

            nodes.add(feedid)
            node_sides.setdefault(feedid, set())

            for t in [userid, authorid, bgm_song_id, bgm_singer_id]:
                nodes.add(t)
                node_sides.setdefault(t, set())

                node_sides[t].add(feedid)
                node_sides[feedid].add(t)

            for t in manual_tag_list.split(";"):
                t = "tag-" + t
                nodes.add(t)
                node_sides.setdefault(t, set())
                node_sides[t].add(feedid)
                node_sides[feedid].add(t)
            for t in manual_keyword_list.split(";"):
                t = "keyword-" + t
                nodes.add(t)
                node_sides.setdefault(t, set())
                node_sides[t].add(feedid)
                node_sides[feedid].add(t)

        os.system("mkdir -p {}".format(save_dir))

        with open(os.path.join(save_dir, "graph.obj"), "wb") as file:
            pickle.dump(nodes, file)
            pickle.dump(node_sides, file)

    if sentences is None:

        print("The number of nodes is: {}".format(len(nodes)))

        pool_num = deepwalk_conf['pool']

        linspace = np.linspace(0, len(nodes), pool_num + 1, dtype=int)

        nodes = list(nodes)

        data_batch = [(nodes[linspace[i]:linspace[i + 1]] * deepwalk_conf['epochs'],
                       node_sides,
                       deepwalk_conf['max_depth'],
                       deepwalk_conf['min_depth']) for i in range(pool_num)]

        pool = Pool(pool_num)

        result = pool.map(walk_path, data_batch)
        pool.close()  # 关闭进程池
        pool.join()  # 阻塞住进程，等待子进程的运行完毕

        sentences = []
        for ret in result:
            sentences.extend(ret)

        random.shuffle(sentences)

        print("Walk Path Examples----")
        for s in sentences[:10]:
            print(s)
            # 验证deep walk路径是否正确
            for i in range(len(s) - 1):
                name1, number1 = s[i].split("-")
                name2, number2 = s[i + 1].split("-")
                if len(data.loc[(data[name1] == int(number1)) & (data[name2] == int(number2))]) > 0:
                    continue
                if name1 == "feedid" or name2 == 'feedid':
                    tmp = data.loc[data['feedid'] == int(number1 if name1 == 'feedid' else number2)]
                    v = tmp[name1 if name1 != 'feedid' else name2].values[0]
                    if (number1 if name1 != 'feedid' else number2) in v.split(";"):
                        continue
                print("Wrong sentence is: {}".format(s))
                print("Wrong path is: {}, {}".format(s[i], s[i + 1]))
                raise ValueError

        with open(os.path.join(save_dir, "walk_path.pkl"), "wb") as file:
            pickle.dump(sentences, file)

    print("The number of walk path is {}".format(len(sentences)))

    word2vec_conf.update({"sentences": sentences})

    model = Word2Vec(**word2vec_conf)

    for name in mapping.keys():
        arr_list = []
        not_train = 0
        for i in range(max(mapping[name].values()) + 1):
            word = name + "-" + str(i)
            if word in model.wv:
                arr_list.append(model.wv[word])
            else:
                arr_list.append(np.random.normal(size=arr_list[-1].shape))
                not_train += 1
        print("Not train number: {}".format(not_train))

        arr = np.array(arr_list)
        np.save(os.path.join(save_dir, name + ".npy"), arr)
        print("The arr of {} is {}".format(name, arr.shape))


def walk_path(paras):
    nodes, node_sides, max_depth, min_depth = paras
    sentences = []
    for node in tqdm(nodes, total=len(nodes)):
        cur = node
        path = []
        while len(path) < max_depth:
            path.append(cur)

            # 选择下一个node
            candidate = node_sides[cur]
            select = random.sample(candidate, 1)[0]
            path.append(select)

            # 选中node如果只有一条边，因为边是双向的，则肯定会返回上一个node
            if len(node_sides[select]) == 1:
                break

            # 使选中的node，不要返回原路node
            select_side = node_sides[select].copy()
            select_side.remove(cur)
            cur = random.sample(select_side, 1)[0]

        if len(path) < min_depth:
            continue
        sentences.append(path[:max_depth])

    return sentences


if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--action_csv', default='action.csv', type=str)
    parser.add_argument('--test_csv', default='test_a.csv|test_b.csv', type=str)
    parser.add_argument('--feed_csv', default='', type=str)
    parser.add_argument('--save_dir', default='feed_emb', type=str)
    parser.add_argument('--map_file', default=None, type=str)
    parser.add_argument('--config_file', default=None, type=str)

    args = parser.parse_args()

    conf = yaml.load(open(args.config_file, "r"))

    run(args.action_csv, args.test_csv, args.feed_csv, args.save_dir,
        conf['Word2Vec'], conf['DeepWalk'],
        args.map_file)
