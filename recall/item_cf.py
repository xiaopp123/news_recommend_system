# -*- coding:utf-8 -*-


"""
参考：
去偏推荐：https://blog.csdn.net/weixin_44922175/article/details/106243500
https://tech.meituan.com/2020/08/20/kdd-cup-debiasing-practice.html
"""

import os
import math
import copy

import numpy as np

from util.data_util import *
from config import *
from collections import defaultdict
from tqdm import tqdm
from util.metric_util import metrics_recall


def item_cf_sim(df, item_created_time_dict):
    """
    item-to-item相似矩阵
    :param df:
    :param item_create_time_dict:
    :return:
    """
    # 1. 用户点击物品列表
    user_item_time_dict = get_user_item_time(df)
    #
    # 共现矩阵
    i2i_sim = {}
    # 物品被多少个用户点击
    item_cnt = defaultdict(int)
    alpha_time_decay = 0.9
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for id_i, (i, i_click_time) in enumerate(item_time_list):
            # 点击物品的用户数
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for id_j, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # 考虑文章正向顺序和反向顺序, 正向
                loc_alpha = 1.0 if id_j > id_i else 0.7
                # 位置权重：类似于点击时间
                loc_weight = loc_alpha * 0.9 ** (abs(id_i - id_j) - 1)
                # 点击时间权重：
                click_time_weight = np.exp(0.7 ** abs(i_click_time - j_click_time))
                # 两篇文章创建时间权重
                created_time_weight = np.exp(0.8 ** abs(item_created_time_dict[i] - item_created_time_dict[j]))

                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += \
                    loc_weight * click_time_weight * created_time_weight / math.log(1 + len(item_time_list))

    # 根据共现矩阵计算相关性矩阵
    alpha = 0.9
    i2i_sim_corr = copy.deepcopy(i2i_sim)
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_corr[i][i] = wij / (math.pow(item_cnt[i], alpha) * math.pow(item_cnt[j], 1 - alpha))
    return i2i_sim_corr


def item_base_recommend(
        user_id, user_item_time_dict, i2i_sim, sim_item_topk,
        recall_item_num, item_topk_click, item_created_time_dict=None,
        emb_i2i_sim=None):
    """
    基于user_id点击过物品，推荐相似的recall_item_num个物品
    :param user_id:
    :param user_item_time_dict:
    :param i2i_sim:
    :param sim_item_topk:
    :param recall_item_num:
    :param item_topk_click:
    :param item_created_time_dict:
    :param emb_i2i_sim:
    :return:
    """
    # 用户历史点击物品列表
    user_hist_item_list = user_item_time_dict[user_id]
    # 推荐物品权重
    item_rank = {}
    for idx, (i, click_time) in enumerate(user_hist_item_list):
        related_item_list = sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)
        for j, wij in related_item_list[:sim_item_topk]:
            if j in user_hist_item_list:
                continue
            # 文章创建时间差权重
            # created_time_weight = np.exp(0.8 ** abs(item_created_time_dict[j] - item_created_time_dict[i]))
            created_time_weight = 1.0
            # 相似文章与历史点击文章位置权重
            # loc_weight = 0.9 * (len(user_hist_item_list) - idx)
            loc_weight = 1.0
            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * wij

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
    # 获取文章属性
    item_df = get_item_info_df(
        file_name=os.path.join(DATA_PATH, 'articles.csv'))
    item_info_list = get_item_info_dict(item_df)
    item_created_time_dict = item_info_list[-1]
    i2i_sim = item_cf_sim(all_click_df, item_created_time_dict)

    # 召回
    hist_click_df, last_click_df = get_hist_and_last_click(all_click_df)
    user_item_time_dict = get_user_item_time(all_click_df)
    # 热门点击
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    user_recall_items_dict = {}
    for user_id in hist_click_df['user_id'].unique():
        user_recall_items_dict[user_id] = item_base_recommend(
            user_id=user_id,
            user_item_time_dict=user_item_time_dict,
            i2i_sim=i2i_sim,
            sim_item_topk=20,
            recall_item_num=10,
            item_topk_click=item_topk_click,
            item_created_time_dict=item_created_time_dict
        )
    print(len(user_recall_items_dict))
    metrics_recall(user_recall_items_dict, last_click_df, topk=50)


if __name__ == '__main__':
    main()
