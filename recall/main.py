# -*- coding:utf-8 -*-

'''
多路召回实践
1. itemcf_sim_itemcf_recall
2. embedding_sim_item_recall
3. youtubednn_recall
4. youtubednn_usercf_recall
5. cold_start_recall
'''


def main():
    user_multi_recall_dict = {
        'itemcf_sim_itemcf_recall': {},
        'embedding_sim_item_recall': {},
        'youtubednn_recall': {},
        'youtubednn_usercf_recall': {},
        'cold_start_recall': {}
    }
    user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict


if __name__ == '__main__':
    main()
