# -*- coding:utf-8 -*-


import os
import math
import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *
import faiss
from util.data_util import *
from collections import defaultdict
from util.metric_util import metrics_recall


def embedding_sim(item_emb_df, save_path, topk=10):
    """
    根据item embedding构建物品相似性列表
    :param item_emb_df:
    :param save_path:
    :param topk:
    :return:
    """
    # embedding 向量索引与物品ID映射表
    item_idx_2_item_id_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    item_emb_col_list = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_col_list].values, dtype=np.float32)
    print(item_emb_np)
    # 单位化的必要性？
    # item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    print(np.linalg.norm(item_emb_np, axis=1, keepdims=True).shape)
    print(sum(item_emb_np[0]))

    # 建立faiss索引
    start_time = time.time()
    # item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    # item_index.add(item_emb_np)
    quantizer = faiss.IndexFlatIP(item_emb_np.shape[1])

    item_index = faiss.IndexIVFFlat(
        quantizer, item_emb_np.shape[1], 100, faiss.METRIC_INNER_PRODUCT)
    item_index.train(item_emb_np)
    item_index.add(item_emb_np)
    print('[DEBUG] faiss build index cost {}'.format(time.time() - start_time))
    start_time = time.time()
    # item_emb_np = item_emb_np[:10000]
    sim_list, idx_list = item_index.search(item_emb_np, topk)
    print('[DEBUG] faiss search cost {}'.format(time.time() - start_time))
    item_sim_dict = defaultdict(dict)

    # 为每个物品构建相似性物品列表
    for target_idx, sim_value_list, relate_idx_list in tqdm(zip(range(len(item_emb_np)), sim_list, idx_list)):
        target_item_id = item_idx_2_item_id_dict[target_idx]
        for relate_idx, sim_value in zip(relate_idx_list[1:], sim_value_list[1:]):
            relate_item_id = item_idx_2_item_id_dict[relate_idx]
            item_sim_dict[target_item_id][relate_item_id] = sim_value
    return item_sim_dict


def item_base_recommend(
        user_id, user_item_time_dict, i2i_sim, sim_item_topk,
        recall_item_num, item_topk_click, item_created_time_dict=None,
        emb_i2i_sim=None):

    # 用户历史点击物品列表
    user_hist_item_list = user_item_time_dict[user_id]

    # 物品被多少个用户点击
    item_cnt = defaultdict(int)
    item_rank = {}
    for idx, (i, click_time) in enumerate(user_hist_item_list):
        related_item_list = sorted(emb_i2i_sim[i].items(), key=lambda x: x[1], reverse=True)
        for j, wij in related_item_list[:sim_item_topk]:
            if j in user_hist_item_list:
                continue
            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for idx, item_id in enumerate(item_topk_click):
            if item_id in item_rank:
                continue
            # 相似性给一个负数
            item_rank[item_id] = - idx - 100
            if len(item_rank) == recall_item_num:
                break
    sorted_item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)
    return sorted_item_rank


def main():
    # 加载训练数据
    all_click_df = get_all_click_sample(
        file_name=os.path.join(DATA_PATH, 'train_click_log.csv'))
    emb_file_name = os.path.join(DATA_PATH, 'articles_emb.csv')
    item_emb_df = pd.read_csv(emb_file_name)
    emb_i2i_sim = embedding_sim(item_emb_df, None, topk=10)
    # print(item_emb_df)

    # 获取文章属性
    item_df = get_item_info_df(
        file_name=os.path.join(DATA_PATH, 'articles.csv'))
    item_info_list = get_item_info_dict(item_df)
    item_created_time_dict = item_info_list[-1]

    # 召回
    hist_click_df, last_click_df = get_hist_and_last_click(all_click_df)
    user_item_time_dict = get_user_item_time(all_click_df)
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    user_recall_items_dict = {}
    for user_id in hist_click_df['user_id'].unique():
        user_recall_items_dict[user_id] = item_base_recommend(
            user_id=user_id,
            user_item_time_dict=user_item_time_dict,
            i2i_sim=None,
            sim_item_topk=20,
            recall_item_num=10,
            item_topk_click=item_topk_click,
            item_created_time_dict=item_created_time_dict,
            emb_i2i_sim=emb_i2i_sim
        )
    metrics_recall(user_recall_items_dict, last_click_df, topk=50)



if __name__ == '__main__':
    main()
