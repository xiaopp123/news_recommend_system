# -*- coding:utf-8 -*-


import os, time, math
import pickle

import pandas as pd
from tqdm import tqdm
from collections import defaultdict


# 读取全量数据
def read_all_data(data_path='../data'):
    train_df = pd.read_csv(data_path + '/train_click_log.csv')
    test_df = pd.read_csv(data_path + '/testA_click_log.csv')
    all_df = train_df.append(test_df)
    print(all_df)
    return all_df


def get_user_item(df):
    print(df.columns)
    df = df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_df = df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_df['user_id'], user_item_df['item_time_list']))
    return user_item_time_dict


def itemcf_sim(df):
    # 1. 获取用户点击物品列表
    user_item_time_dict = get_user_item(df)
    print(user_item_time_dict)
    # 2. 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user_id, item_time_list in tqdm(user_item_time_dict.items()):
        for article_id, click_time in item_time_list:
            i2i_sim.setdefault(article_id, {})
            item_cnt[article_id] += 1
            for to_article_id, to_click_time in item_time_list:
                if article_id == to_article_id:
                    continue
                i2i_sim[article_id].setdefault(to_article_id, 0)
                i2i_sim[article_id][to_article_id] += 1 / math.log(len(item_time_list) + 1)
    for article_id, related_item in i2i_sim.items():
        for to_article_id, weight in related_item.items():
            i2i_sim[article_id][to_article_id] = weight / math.sqrt(item_cnt[article_id] * item_cnt[to_article_id])
    pickle.dump(i2i_sim, open('../data/item_cf_i2i_sim.pkl', 'wb'))
    return i2i_sim


def get_item_topk_click(df, k):
    topk_click = df['click_article_id'].value_counts().index[:k]
    return topk_click


def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    # 1. 获取user_id历史点击物品
    user_hist_item_list = user_item_time_dict[user_id]
    # 2.
    item_rank = {}
    for i, (article_id, click_time) in enumerate(user_hist_item_list):
        sorted_related_item = sorted(i2i_sim[article_id].items(), key=lambda x: x[1], reverse=True)
        for to_article, weight in sorted_related_item[:sim_item_topk]:
            if to_article in user_hist_item_list:
                continue
            item_rank.setdefault(to_article, 0)
            item_rank[to_article] += weight

    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank:
                continue
            item_rank[item] = -i - 100
            if len(item_rank) == recall_item_num:
                break
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
    return item_rank


all_data_df = read_all_data()


def train():
    all_data_df = read_all_data()
    i2i_sim = itemcf_sim(all_data_df)



def test():
    i2i_sim = pickle.load(open('../data/item_cf_i2i_sim.pkl', 'rb'))
    print(i2i_sim)
    item_topk_click = get_item_topk_click(all_data_df, k=50)
    user_item_time_dict = get_user_item(all_data_df)
    # 相似文章的数量
    sim_item_topk = 10

    # 召回文章数量
    recall_item_num = 10

    user_recall_items_dict = {}
    cnt = 0
    start_time = time.time()
    for user_id in tqdm(all_data_df['user_id'].unique()):
        user_recall_items_dict[user_id] = item_based_recommend(
            user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click)

    print(user_recall_items_dict)



def main():
    test()



if __name__ == '__main__':
    main()
