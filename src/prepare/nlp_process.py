import pandas as pd
import random
import os
import pickle
import numpy as np
from gensim.models.word2vec import Word2Vec


def run(feed_csv, word_dim, workers, map_path, save_path):
    unk_num = 20000

    feed_info = pd.read_csv(feed_csv)
    feed_info['description'] = feed_info['description'].fillna("-1")
    feed_info['ocr'] = feed_info['ocr'].fillna("-1")
    feed_info['asr'] = feed_info['asr'].fillna("-1")

    map_id = {}

    corpus = []

    corpus.extend(feed_info['ocr'].values.tolist())
    corpus.extend(feed_info['asr'].values.tolist())
    corpus.extend(feed_info['description'].values.tolist())

    # 计算每个单词的频次
    for text in corpus:
        for v in str(text).split(" "):
            if v in map_id:
                map_id[v] += 1
            else:
                map_id[v] = 1
    print("The number of origin map_id is: {}".format(len(map_id)))
    # 根据频次进行排序
    word_count = sorted([(word, num) for word, num in map_id.items()], key=lambda x: x[1])

    # 频次低的设为同一类：unk
    map_id = {w[0]: 0 for w in word_count[:unk_num]}
    map_id.update({w[0]: i + 1 for i, w in enumerate(word_count[unk_num:])})
    word_not_unk = [w[0] for w in word_count[unk_num:]]

    print("The number of map_id is: {}".format(len(set(map_id.values()))))

    # 将数据中的低频次单词置换为unk
    sentences = []
    for text in corpus:
        tmp = []
        for t in str(text).split(" "):
            if map_id[t] == 0:
                tmp.append("unk")
            else:
                tmp.append(t)
        sentences.append(tmp)

    print("The number of corpus is: {}".format(len(sentences)))
    print(sentences[:10])

    random.shuffle(sentences)

    model = Word2Vec(sentences=sentences, window=10, size=word_dim, min_count=1, workers=workers, iter=10, sg=1)
    # model = Word2Vec(sentences=sentences, window=10, vector_size=word_dim, min_count=1, workers=workers, epochs=20, sg=1)

    # 获取对应顺序的向量
    word_arr = [model.wv['unk']]
    for word in word_not_unk:
        word_arr.append(model.wv[word])
    word_arr = np.array(word_arr)

    if os.path.exists(map_path):
        with open(map_path, 'rb') as file:
            all_map_id = pickle.load(file)
    else:
        all_map_id = {}

    all_map_id.update({"word": map_id})

    with open(map_path, 'wb') as file:
        pickle.dump(all_map_id, file)

    print("The arr shape is: {}".format(word_arr.shape))
    np.save(save_path, word_arr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--feed_csv', default='feed_info.csv', type=str)
    parser.add_argument('--word_dim', default=256, type=int)
    parser.add_argument('--workers', default=20, type=int)
    parser.add_argument('--map_path', default='', type=str)
    parser.add_argument('--save_path', default='', type=str)

    args = parser.parse_args()

    run(args.feed_csv, args.word_dim, args.workers, args.map_path, args.save_path)
