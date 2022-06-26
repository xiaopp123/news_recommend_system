# -*- coding:utf-8 -*-
import os

import numpy as np
import pandas as pd
from config import *


# 采样数据,
def get_all_click_sample(file_name, sample_cnt=10000):
    """
    从全量数据集中采样sample_cnt个用户记录
    :param file_name:
    :param sample_cnt:
    :return:
    """
    data_df = pd.read_csv(file_name)
    all_uid_list = data_df.user_id.unique()
    sample_uid_list = np.random.choice(all_uid_list, sample_cnt)
    sample_data_df = data_df[data_df['user_id'].isin(sample_uid_list)].\
        drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    return sample_data_df


def get_user_item_time(df):
    """
    根据点击时间获取用户点击物品列表
    :param df:
    :return:
        {user_id: {item_1: time_1, item_2: time_2}}
    """
    # 按照点击时间排序
    sorted_df = df.sort_values('click_timestamp')

    def make_item_time_pair(x):
        return list(zip(x['click_article_id'], x['click_timestamp']))
    # 按照user_id进行聚合，对同一用户下的点击文章和时间构成pair对列表
    user_item_time_df = sorted_df.groupby('user_id')['click_article_id', 'click_timestamp'].\
        apply(lambda x: make_item_time_pair(x)).\
        reset_index().\
        rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k=50):
    """获取近期点击最多的文章
    :param click_df:
    :param k:
    :return:
    """
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def get_hist_and_last_click(df):
    """
    获取用户历史点击物品和最后一次点击物品
    :param df:
    :return:
    """
    sorted_df = df.sort_values(by=['user_id', 'click_timestamp'])
    # 每个用户最后一次点击
    last_df = sorted_df.groupby('user_id').tail(1)

    # 如果用户只有一个点击时，hist和最后一个点击均为该物品, 允许信息泄漏
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    hist_df = sorted_df.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return last_df, hist_df


def get_item_info_df(file_name):
    """
    读取点击物品文件
    :param file_name:
    :return:
    """
    item_info_df = pd.read_csv(file_name)
    item_info_df = item_info_df.rename(
        columns={'article_id': 'click_article_id'})
    return item_info_df


def get_item_info_dict(item_info_df):
    """
    获取文章创建日期
    :param item_info_df:
    :return:
    """
    # 文章创建时间戳归一化处理
    max_min_scaler = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)
    # 文章类型映射表
    # 文章字数映射表
    # 文章创建时间映射表
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    return None, None, item_created_time_dict


def main():
    file_name = os.path.join(DATA_PATH, 'train_click_log.csv')
    get_all_click_sample(file_name)
    # 测试读取文章
    item_file_name = os.path.join(DATA_PATH, 'articles.csv')
    item_df = get_item_info_df(item_file_name)
    print(item_df.info)
    # 获取文章基本属性
    item_info_list = get_item_info_dict(item_df)
    print(item_info_list[-1])


if __name__ == '__main__':
    main()
