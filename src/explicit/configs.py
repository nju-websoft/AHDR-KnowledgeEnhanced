common_dict_config = {
    'link_method': 'FALCON2', # 'TAGME|FALCON2|REL'
    'embedding_method': 'wikidata_rdf2vec_sg_200_1',  # 'wiki2vec|wikidata_rdf2vec_sg_200_1|kgtk',
    'test_collection_name': 'ACORDAR', # 'ACORDAR|NTCIR-E|ntcir16'
    'train_or_test': 'test',
    'data_field': 'metadata', 
    'res_dir_name': 'multi_run',
    'k': 10, # rerank top k
}

sparse_methods = ['BM25']
method_suffix = ' [m]'
sparse_methods = [method + method_suffix for method in sparse_methods]


# for retrieve 
retrieve_dict_config = {
    'dim': 200, # 200
    'stg': 'rerank', # [rerank|total] indicates where rerank or retrieve, common choice is `rerank`
     # test 
    'use_entity_matching_confidence': False, # True
    'margin1': 0, # only used when use_entity_matching_confidence
    'use_query_score_confidence': False, # True
    'margin2': 0, # only used when use_query_score_confidence
    'filter_rho': False, # True # indicate whether to filter rho in retrieve process
    'do_entity_filter': False, # True # indicate whether to filter entity in retrieve process
    'filter_link_pb_lb': 0, 
    'filter_link_pb_ub': 1,
    'filter_rho_lb': 0,
    'filter_rho_ub': 1,
    'qd_ent_weight': 0.5, # 0 ~ 1
    'rho_score_op': 0, # deprecated
    'ent_word_sim_op': 2,
    'word_sim_op': 2,
    'qd_ent_score_op': 2,
    'qd_word_score_op': 2,
    'use_top_link_pb_query': False,
    'use_expanded_query': False,
    'use_REL_query': False, # True,
    'use_best_rho_query': False,
    'use_expanded_entity_query': False,
    'use_ngram_query': False,
    'use_tfidf_style_method': False,
    'use_cos_score_entity_query': False,
    'add_ngram_cos_weight': False,
    'drop_last_entity': False,
    'use_word_embedding': True,
    'word_embedding_use_way': 'multi_level2', # 'concate|one|multi_level1|multi_level2|full'
    'use_entity_embedding': True,
    'entity_sim_df_lb': 0, # deprecated
    'entity_sim_tf_lb': 0, # deprecated
}
"""
score_op notes:
1: 2-stage avg: (sum1/len1 + sum2/len2) / 2
2: harmonic avg: 2 * sum1 * sum2 / (sum1 + sum2)
3: total avg: sum1 + sum2 / len1 + len2
4: Geometric avg: sqrt(sum1 * sum2)
5: max: max(sum1, sum2)
6: min: min(sum1, sum2)
"""

# for evaluate
eval_dict_config = {
    'do_filter': False, # True    # indicate whehter do_filter in eval process
    'link_pb_lb': 0,
    'link_pb_ub': 1, 
    'rho_lb': 0,  
    'rho_ub': 1, 
    'len_one_check': False, # False,
    'metadata_tokens_len_lb': 0, 
    'filtered_1_start_query': False, # False,
    'query_len_ub': 100,
    'use_fasttext_filter': False,
    'use_origin_score': False, # indicate where use orgin score (sparse score) or not
}
s = 'abc'
s.startswith
indicate_run_id = 0

def check_retrieve_config(retrieve_dict_config):
    assert retrieve_dict_config['do_entity_filter'] or retrieve_dict_config['filter_link_pb_lb'] == 0 and retrieve_dict_config['filter_link_pb_ub'] == 1 and retrieve_dict_config['filter_rho_lb'] == 0 and retrieve_dict_config['filter_rho_ub'] == 1, 'filter_link_pb_lb and filter_link_pb_ub should be 0 and 1 when do_entity_filter is False'

    assert retrieve_dict_config['use_entity_matching_confidence'] or retrieve_dict_config['margin1'] == 0, 'margin1 should be 0 when use_entity_matching_confidence is False'

    assert retrieve_dict_config['use_query_score_confidence'] or retrieve_dict_config['margin2'] == 0, 'margin2 should be 0 when use_query_score_confidence is False'

    assert retrieve_dict_config['use_top_link_pb_query'] + retrieve_dict_config['use_expanded_query'] + retrieve_dict_config['use_REL_query'] + retrieve_dict_config['use_best_rho_query'] + retrieve_dict_config['use_expanded_entity_query'] + retrieve_dict_config['use_ngram_query'] + retrieve_dict_config['use_cos_score_entity_query'] + retrieve_dict_config['use_word_embedding'] <= 1, 'only one of use_top_link_pb_query, use_expanded_query, use_REL_query, use_best_rho_query, use_expanded_entity_query, use_cos_score_entity_query, use_word_embedding can be True'

    assert retrieve_dict_config['drop_last_entity'] == 0 or retrieve_dict_config['use_cos_score_entity_query'] is True, 'if drop_last_entity != 0, then use_cos_score_entity_query should be True'

    assert retrieve_dict_config['add_ngram_cos_weight'] == False or retrieve_dict_config['use_cos_score_entity_query'] is True, 'if add_ngram_cos_weight == True, then use_cos_score_entity_query should be True'

    assert retrieve_dict_config['use_entity_matching_confidence'] or retrieve_dict_config['rho_score_op'] == 0, 'if not use_entity_matching_confidence, then rho_score_op should be 0'


def check_eval_config(eval_dict_config):
    assert eval_dict_config['do_filter'] or eval_dict_config['link_pb_lb'] == 0 and eval_dict_config['link_pb_ub'] == 1 and eval_dict_config['rho_lb'] == 0 and eval_dict_config['rho_ub'] == 1, 'link_pb_lb and link_pb_ub should be 0 and 1 when do_filter is False'

    assert eval_dict_config['do_filter'] or eval_dict_config['len_one_check'] == False, 'len_one_check should be False when do_filter is False'

check_retrieve_config(retrieve_dict_config)
check_eval_config(eval_dict_config)

class ObjectDict(dict):
    """ allows object style access for dictionaries """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError('No such attribute: %s' % name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError('No such attribute: %s' % name)

retrieve_dict_config.update(common_dict_config)
eval_dict_config.update(common_dict_config)

common_config = ObjectDict(common_dict_config)
retrieve_config = ObjectDict(retrieve_dict_config)
eval_config = ObjectDict(eval_dict_config)

if common_config.test_collection_name == 'ACORDAR':
    topic_link_res_path = f'/home/yourname/code/erm/data/topic_link_results/{common_config.test_collection_name}_{common_config.train_or_test}_topic_link_result.json' 
    if retrieve_config.use_top_link_pb_query: topic_link_res_path = topic_link_res_path.replace('.json', '_top_link_pb.json')
    elif retrieve_config.use_expanded_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_expanded_query_id_link_result.json'
    elif retrieve_config.use_REL_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_query_link_result_REL.json'
    elif retrieve_config.use_best_rho_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_test_topic_link_result_best_rho.json'
    elif retrieve_config.use_expanded_entity_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_test_topic_link_result_expanded_entities.json'
    elif retrieve_config.use_ngram_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_test_query_link_result_ngram_merged.json'
    elif retrieve_config.use_cos_score_entity_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ACORDAR_test_query_link_result_with_cos_score.json' # f'/home/yourname/code/erm/data/topic_link_results/ACORDAR_test_query_link_result_ngram_with_cos_score.json'
else: 
    if retrieve_config.use_ngram_query:
        topic_link_res_path = '/home/yourname/code/erm/data/topic_link_results/ntcir_test_query_link_result_ngram_merged.json'
    else:
        topic_link_res_path = f'/home/yourname/code/erm/data/topic_link_results/ntcir_{common_config.train_or_test}_topic_link_result.json'

def format_mapping(format):
    return format
