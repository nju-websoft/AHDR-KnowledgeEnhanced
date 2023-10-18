import pytrec_eval
import json
import os
from database import sql_one, sql_list
import numpy as np
from typing import List
from util import json_load, json_dump, get_max_ind, get_metadata_dict
from configs import eval_config as config, eval_dict_config, sparse_methods, indicate_run_id, topic_link_res_path

rho_lb = config.rho_lb
rho_ub = config.rho_ub
link_pb_lb = config.link_pb_lb
link_pb_ub = config.link_pb_ub

print(config)

qrels_path = f'/home/yourname/code/erm/data/qrels/{config.test_collection_name}_{config.train_or_test}_qrels.json' # '/home/yourname/code/erm/file/qrels_train.json'
with open(qrels_path, 'r') as fp:
    qrels = json.load(fp)

def simple_eval(res_dict, k, ground_truth_qrels = qrels):
    """
    res_dict format:
    key: query_id
    value: list-type dataset_ids
    """
    # res_dict = json_load(res_dict_path)
    evaluator = pytrec_eval.RelevanceEvaluator(ground_truth_qrels, {'map_cut.5','map_cut.10','ndcg_cut.5','ndcg_cut.10'})
    for one_res in res_dict.values():
        one_res_bak = one_res.copy()
        one_res.sort(key=lambda x: x[1], reverse=True)
        # assert one_res == one_res_bak
    run = dict([(query_id, dict((str(did), score) for did, score in did_scores[:k])) for (query_id, did_scores) in res_dict.items()])
    rank_metric_val = evaluator.evaluate(run)
    mean_dict = {}
    for metric_name in ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_5', 'map_cut_10']:
        mean_dict[metric_name+'_mean'] = get_mean([val[metric_name] for val in rank_metric_val.values()])
        
    if config.test_collection_name == 'NTCIR-E':
        assert len(rank_metric_val) == 96
        mean_all_dict = mean_dict
    elif config.test_collection_name == 'ntcir16':
        assert len(rank_metric_val) == 58
        mean_all_dict = mean_dict
    else:
        assert config.test_collection_name == 'ACORDAR'
        if config.data_field == 'metadata':
            assert len(rank_metric_val) == 483
            mean_all_dict = {}
            for k, v in mean_dict.items():
                mean_all_dict[k] =  np.around(v * 483 / 493, 4)
        else:
            assert len(rank_metric_val) == 493
            mean_all_dict = mean_dict

    output_res = {
        'eval_config': eval_dict_config,
        'each query': rank_metric_val,
        'mean': mean_dict,
        'mean_all': mean_all_dict
    }

    return output_res
    # json_dump(output_res, out_path)

def get_mean(data: List):
    return np.around(np.mean(data), 4)

query_link_res = json_load(topic_link_res_path)
id2name = json_load(f'/home/yourname/code/erm/data/topic_link_results/id2name_{config.test_collection_name}_{config.train_or_test}.json')
def get_prob(query_id):
    if query_id not in query_link_res.keys() or len(query_link_res[query_id]) == 0: return 0
    probs = [anno['link_probability'] for anno in query_link_res[query_id]]
    return max(probs)

metadata_dict = get_metadata_dict()
def get_metadata_tokens_len(did):
    return len(metadata_dict[did].split(' '))

predict_res_path = '/home/yourname/code/fastText/pred.json' # '/home/yourname/code/fastText/data/results/predictions.json' # '/home/yourname/code/erm/src/binary_classification/predict_res.json'
predict_res = json_load(predict_res_path) if config.use_fasttext_filter else None

def gen_rerank_ids(dense_res, candidate_res, k = 10):
    global rho_lb, rho_ub, link_pb_lb, link_pb_ub
 
    rerank_dict = dense_res
    all_query_res = {}
    filtered_query_cnt = 0
    for query_id in candidate_res.keys():

        candidate_topk = candidate_res[query_id][:k]
        # print(query_id)
        one_query_res = rerank_dict.get(query_id, None)
        # print('one_query_res', one_query_res)
        if one_query_res is None: 
            one_query_res = candidate_topk

        if config.use_fasttext_filter and predict_res.get(id2name[query_id], 0) == 0: one_query_res = []

        if config.do_filter:
            
            flag = False
            link_res_total = query_link_res.get(query_id, [])
            if link_res_total is None: continue
            # link_res_total = []
            
            link_res = [] # link_res_total
            for anno in link_res_total:
                if rho_lb <= anno['rho'] <= rho_ub: link_res.append(anno)
            length = len(link_res)
            
            if config.use_fasttext_filter and predict_res.get(id2name[query_id], 0) == 0:
                flag = True
            elif config.filtered_1_start_query and int(query_id) > 1000:
                flag = True
            elif len(id2name[query_id].split(' ')) > config.query_len_ub:
                flag = True
            elif length == 0:
                flag = True 
            elif config.len_one_check and length != 1: 
                flag = True
            else:
                rho = sum(item['rho'] for item in link_res) / length
                link_pb = sum(item['link_probability'] for item in link_res) / length
                if not rho_lb <= rho < rho_ub \
                    or not link_pb_lb <= link_pb <= link_pb_ub: 
                    flag = True
            if flag:
                filtered_query_cnt += 1
                one_query_res = candidate_topk


        id2score = {}
        for (id, score) in one_query_res: id2score[id] = score
        
        rerank_id2score = {}

        # style 1: use dense score
        if not config.use_origin_score:
            for (did, sparse_score) in candidate_topk:
                # assert did in id2score.keys()
                if did in id2score.keys():
                    rerank_id2score[did] = id2score[did]
                else: 
                    rerank_id2score[did] = 0
                    # print('did', did, 'not in id2score.keys()')


        # style 2: use sparse score
        else:
            hit_dids = []
            hit_scores = []
            for (did, score) in candidate_topk:
                rerank_id2score[did] = score

                # if get_metadata_tokens_len(did) < config.metadata_tokens_len_lb: continue

                if did in id2score.keys():
                    hit_dids.append(did)
                    hit_scores.append(score)

            """
            origin: 4 5 7 8
            dense: 2 4 1 2
            rerank: 7 4 8 5
            """

            hit_dids.sort(key=lambda x: (id2score[x], rerank_id2score[x]), reverse=True)
            for i in range(len(hit_dids)):
                rerank_id2score[hit_dids[i]] = hit_scores[i]

        all_query_res[query_id] = sorted(list(rerank_id2score.items()), key=lambda x: x[1], reverse=True)

    print('filtered_query_cnt', filtered_query_cnt)

    return all_query_res






def eval_merged(type, ind, multi_run_or_signle = 'multi_run'):
    
    if multi_run_or_signle == 'multi_run':
        multi_run_suffix = f'22/{ind}' if config.test_collection_name == 'ntcir' else f'36/{ind}'
        merged_dir_perfix = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/metadata/multi_run/{multi_run_suffix}'
    else:
        merged_dir_perfix = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/metadata/single_run/{ind}'
    
    for method in sparse_methods:
        if type == 'sparse':
            tail = f'with_sparse_merged'
        elif set(type) == set('wes'):
            tail = f'3_score_merged'
        elif type == 'irsota':
            tail = 'with_ir_sota_merged'
        elif type == 'two':
            tail = f'merged'
        else:
            tail = f'{type}_score_merged'
        # tail = f'{type}_score_merged' if type != 'sparse' else f'with_sparse_merged'
        merged_dir_path = f'{merged_dir_perfix}/{method}/{tail}'
        target_fnames = list(filter(lambda x: x.endswith('_merged.json'), os.listdir(merged_dir_path)))
        for fname in sorted(target_fnames):
            if fname.endswith('_eval.json'): continue
            merged_res_path = merged_dir_path + '/' + fname            
            merged_res = json_load(merged_res_path)['result']
            
            eval_res = simple_eval(merged_res, config.k)
            print(f"method: {method}, ind: {ind}, {fname} {eval_res['mean_all']}")
            eval_path = merged_res_path.replace('.json', f'_eval.json')
            json_dump(eval_res, eval_path)



if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'tmp':
        tmp_eval()
    elif sys.argv[1] == 'print_ndcg':
        print_with_sparse_ndcg()
    elif sys.argv[1] == 'single':
        eval_keds()
    elif sys.argv[1] == 'multi_run':
        if config.test_collection_name == 'ntcir':
            eval_multi_run()
            get_best_ndcg5()
        else:
            eval_multi_run()
            get_best_ndcg5()
            # merge_folds_result()
    elif sys.argv[1] == 'dpr':
        eval_dpr_res()
    elif sys.argv[1] == 'bert':
        eval_bert_reranker_result()
    elif sys.argv[1] == 'merge':
        eval_merge() 
    else:
        # stupid
        assert sys.argv[1] == 'merge_we' or sys.argv[1] == 'merge_ws' or sys.argv[1] == 'merge_es' \
            or sys.argv[1] == 'merge_wes' or sys.argv[1] == 'merge_sparse' or sys.argv[1] == 'merge_two' \
            or sys.argv[1] == 'merge_irsota', 'wrong argv'
        
        eval_merged(sys.argv[1].split('_')[-1], int(sys.argv[2]), 'single')


