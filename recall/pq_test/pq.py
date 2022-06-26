# -*- coding:utf-8 -*-


import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.spatial.distance import cdist


def train(vec, M, Ks):
    """
    :param vec: 向量
    :param M: 子向量组数
    :param Ks: 每组向量聚类个数
    :return: codeword: [M, Ks, Ds]，
        codeword[m][k]表示第m组子向量第k个子向量所属的聚类中心向量
    """
    Ds = int(vec.shape[1] / M)
    codeword = np.empty((M, Ks, Ds), np.float32)

    for m in range(M):
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # 第m组子向量vec_sub聚成Ks类
        # kmeans2返回两个结果，第一个是原始向量归属类目的中心向量，第二个是类目ID
        codeword[m], label = kmeans2(vec_sub, Ks)
    return codeword


def encode(codeword, vec):
    """
    :param codeword: 码本，shape为[M, Ks, Ds]
    :param vec: 原始向量
    :return: pqcode: pq编码结果,
        shape为[N, M]，每个原始向量用M组子向量的聚类中心ID表示
    """
    M, Ks, Ds = codeword.shape
    # pq编码shape为[N, M]
    pqcode = np.empty((vec.shape[0], M), np.int64)
    for m in range(M):
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        # 第m组子向量
        # 第m组子向量中每个子向量在第m个码本中查找距离最近的
        pqcode[:, m], dist = vq(vec_sub, codeword[m])
    return pqcode


def search(codeword, pqcode, query):
    """
    :param codeword:
    :param pqcode: pq编码结果, shape为[N, M]，每个原始向量用M组子向量的聚类中心ID表示
    :param query: 查询向量[1, d]
    :return: dist：查询向量与原始向量的距离，shape为[N,]
    """
    M, Ks, Ds = codeword.shape
    # 距离向量表, [M, Ks]
    dist_table = np.empty((M, Ks))
    for m in range(M):
        query_sub = query[m * Ds: (m + 1) * Ds]
        # query_sub向量与第m个码本每个向量距离
        dist_table[m, :] = cdist([query_sub], codeword[m], 'sqeuclidean')[0]

    # dist_table[range(M), pqcode] 为 query向量与原始向量在每个子向量的聚类，shape为[N, M]
    # 每组子向量距离相加
    dist = np.sum(dist_table[range(M), pqcode], axis=1)
    return dist


def main():
    # 数据量
    N = 50000
    # 向量维度
    d = 128
    # 每组子向量聚类个数
    Ks = 32
    # 训练向量[N, d]
    vec_train = np.random.random((N, d))
    # 查询向量[1, d]
    # mock 第100个是距离查询向量最近的
    selected_vec = vec_train[100]
    query_vec = selected_vec + [np.random.uniform(-0.001, 0.001) for _ in range(d)]
    query = np.random.random((1, d))
    # 子向量组数
    M = 4

    # 对原始向量划分子向量组，并对每组子向量进行聚类
    codeword = train(vec_train, M, Ks)
    # pq编码
    pqcode = encode(codeword, vec_train)
    # 查询向量
    dist = search(codeword, pqcode, query_vec)

    sorted_dist = sorted(enumerate(dist), key=lambda x: x[1])
    print(sorted_dist[0])
    """
    (100, 6.850794458722508)
    """


if __name__ == '__main__':
    main()


"""
reference:
[1]. https://github.com/matsui528/nanopq/blob/main/nanopq/pq.py
[2]. https://www.jstage.jst.go.jp/article/mta/6/1/6_2/_pdf
"""
