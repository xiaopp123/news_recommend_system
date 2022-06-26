# -*- coding:utf-8 -*-


def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
    依次评估召回的前10，20，30，40，...，文章的命中率
    :param user_recall_items_dict:
    :param trn_last_click_df:
    :param topk:
    :return:
    """
    last_click_item_dict = dict(
        zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            tmp_recall_items = [x[0] for x in item_list[:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print('top {}, hit number: {}, user num: {}, hit rate: {}'.format(k, hit_num, user_num, hit_rate))


def main():
    pass


if __name__ == '__main__':
    main()
