# -*- coding:utf-8 -*-


import faiss
import numpy as np


def test_IndexFlatL2(vec_train, query, top_k=5):
    """
    暴力检索
    :param vec_train:
    :param query:
    :return:
    """
    N, d = vec_train.shape
    # 1. 创建索引
    index = faiss.IndexFlatL2(d)
    # 2. 添加数据集
    index.add(vec_train)
    # 3. 检索
    dist_list, label_list = index.search(np.array([query]), k=top_k)
    print(dist_list, label_list)


def test_IndexIVFFlat(vec_train, query_vec, top_k=5):
    """
    通过创建倒排索引优化
    流程：
    使用k-means对train向量进行聚类，查询时query_vec所归属的类目中进行检索
    :param vec_train:
    :param query_vec:
    :param top_k:
    :return:
    """
    nlist = 100  # 聚类中心的个数
    N, d = vec_train.shape
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 添加 训练集
    index.train(vec_train)
    index.add(vec_train)
    # 检索
    res = index.search(query_vec, k=top_k)
    print(res)


def main():
    # 数据量
    N = 50000
    # 向量维度
    d = 128
    vec_train = np.ascontiguousarray(np.random.random((N, d)), np.float32)

    # mock 第100个是距离查询向量最近的
    selected_vec = vec_train[100]
    query_vec = selected_vec + [np.random.uniform(-0.001, 0.001) for _ in range(d)]
    query_vec = np.ascontiguousarray(query_vec, np.float32)
    # 1. 暴力检索，全量检索
    # test_IndexFlatL2(vec_train, query_vec)
    # 2. 倒排索引
    test_IndexIVFFlat(vec_train, query_vec)


if __name__ == '__main__':
    main()
