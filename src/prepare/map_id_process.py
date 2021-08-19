import pandas as pd
import pickle
import os


def run(action_csv, feed_csv, save_path):
    mask_num = {"feedid": 5000,
                "authorid": 3000,
                "bgm_song_id": 5000,
                "bgm_singer_id": 3000,
                "keyword": 5000,
                "userid": 10000}

    action = pd.DataFrame()
    for file in action_csv.split(","):
        action = action.append(pd.read_csv(file))
    feed_df = pd.read_csv(feed_csv)

    for name in ['authorid', 'bgm_song_id', 'bgm_singer_id']:
        feed_df[name] = feed_df[name].fillna(-1)
    feed_df['manual_tag_list'] = feed_df['manual_tag_list'].fillna("-1")
    feed_df['manual_keyword_list'] = feed_df['manual_keyword_list'].fillna("-1")

    train_df = pd.merge(action, feed_df, on='feedid')

    diff = {}
    for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']:
        train = set(train_df[name].drop_duplicates().values.tolist())
        test = set(feed_df[name].drop_duplicates().values.tolist())
        diff[name] = set.difference(test, train)
        print("训练集未出现的{} 数量为: {}".format(name, len(diff[name])))
        # 将出现次数较少的也置为unk
        diff[name].update(
            set(train_df.groupby(name, as_index=False)['userid'].count().sort_values(
                by='userid')[name].values[:mask_num[name]])
        )

    print("mask后测试集未在训练集出现的：{}".format({key: len(value) for key, value in diff.items()}))

    result = {}

    if mask_num['userid'] > 0:
        user_count = action.groupby('userid', as_index=False)['feedid'].count().sort_values(
            by='feedid')['userid'].values
        user_map = {uid: 0 for uid in user_count[:mask_num['userid']]}
        for i, uid in enumerate(user_count[mask_num['userid']:]):
            user_map[uid] = i + 1
    else:
        user_map = {uid: i for i, uid in enumerate(train_df['userid'].drop_duplicates().values)}

    result['userid'] = user_map

    for name in ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']:
        d = {0: 0}
        for v in feed_df[name].drop_duplicates().values:
            if d.get(v) is not None:
                continue
            if v in diff[name]:
                d[v] = 0
            else:
                d[v] = max(d.values()) + 1
        result[name] = d

    d = {}
    for value in feed_df['manual_tag_list'].values:
        for v in value.split(";"):
            if d.get(v) is None:
                d[v] = len(d)
    result['tag'] = d

    count = {}
    for value in feed_df['manual_keyword_list'].values:
        for v in value.split(";"):
            if v in count:
                count[v] += 1
            else:
                count[v] = 1

    unk_num = mask_num['keyword']
    # 根据频次进行排序
    count = sorted([(word, num) for word, num in count.items()], key=lambda x: x[1])
    # 频次低的设为同一类：unk
    d = {w[0]: 0 for w in count[:unk_num]}
    d.update({w[0]: i + 1 for i, w in enumerate(count[unk_num:])})

    result['keyword'] = d

    print("所有category数量分布:")
    for name, one_map in result.items():
        assert len(set(one_map.values())) == max(one_map.values()) + 1, \
            "The map of {} is wrong. length is {}, max_id is {}".format(
                name, len(set(one_map.values())), max(one_map.values()))
        print(name, len(set(one_map.values())))

    if os.path.exists(save_path):
        with open(save_path, 'rb') as file:
            final_map = pickle.load(file)
    else:
        final_map = {}

    final_map.update(result)

    os.system("mkdir -p {}".format(os.path.dirname(save_path)))

    with open(save_path, 'wb') as file:
        pickle.dump(final_map, file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--action_csv', default='feed_embeddings.csv', type=str)
    parser.add_argument('--feed_csv', default='feed_info.csv', type=str)
    parser.add_argument('--save_path', default='map.pkl', type=str)

    args = parser.parse_args()

    run(args.action_csv, args.feed_csv, args.save_path)
