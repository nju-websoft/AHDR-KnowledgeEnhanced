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

def eval_sparse():
    output_list = []

    for (query_id, res) in sparse_res.items():
        print(query_id)

        ent_one_res = ent_res.get(query_id, [])

        res_dataset_ids_ = []
        pred_scores = []
        sz = len(res)
        for i, did in enumerate(res[:10]):
            res_dataset_ids_.append(did)
            # pred_scores.append(sz - i)
        
        
        res_dataset_ids = sorted(res_dataset_ids_, key = lambda x: -ent_one_res.index(x) + 100 if x in ent_one_res else -res_dataset_ids_.index(x), reverse=True)
        
        pred_scores = list(range(10, 0, -1))

        top_10_ids = [str(_) for _ in res_dataset_ids[:10]]

        pools = set(qrels[query_id].keys())
        pool_len = len(set(top_10_ids) & pools)

        output_list.append({
            'query_id': query_id,
            'res_dataset_ids': res_dataset_ids, # [_[0] for _ in one_rec],
            'pred_scores': pred_scores, # [_[1] for _ in one_rec]
            'pool_len': pool_len,
        })
    
    cal_ndcg_val(output_list)

    np.set_printoptions(precision = 4)
    output_list.append(
        {
            'NDCG_at_5_mean': get_mean([item['NDCG_at_5_10'][0] for item in output_list]),
            'NDCG_at_10_mean': get_mean([item['NDCG_at_5_10'][1] for item in output_list]),
            'MAP_at_5_mean': get_mean([item['MAP_at_5_10'][0] for item in output_list]),
            'MAP_at_10_mean': get_mean([item['MAP_at_5_10'][1] for item in output_list]),
            'mean_pool_len': get_mean([item['pool_len'] for item in output_list]),
        }
    )
    outpath = 'orge_res_eval_rerank.json'
    with open(outpath, 'w+') as fpw:
        json.dump(output_list, fpw, indent=2)

# # def eval(dense_res_path,  candidate_res_path, k):
# def eval(dense_res,  candidate_res, k):
#     rerank_res = gen_rerank_ids(dense_res, candidate_res, k) # gen_rerank_ids(dense_res_path, candidate_res_path, k)
#     eval_res = simple_eval(rerank_res, k) # simple_eval(rerank_path, eval_path)
#     return rerank_res, eval_res

def get_rg(n_intervel):
    link_pbs = []
    rhos = []
    for val in query_link_res.values():
        if val is None: continue
        length = len(val)
        if length == 0: continue
        rho = sum(item['rho'] for item in val) / length
        link_pb = sum(item['link_probability'] for item in val) / length
        link_pbs.append(link_pb)
        rhos.append(rho)
    link_pbs.sort()
    rhos.sort()
    length = len(link_pbs)
    step = len(link_pbs) // (n_intervel - 1)
    rg1 = [link_pbs[i] for i in range(0, length, step)] 
    rg2 = [rhos[i] for i in range(0, length, step)]
    if len(rg1) < n_intervel:
        rg1.append(link_pbs[-1])
        rg2.append(rhos[-1])
    return rg1, rg2

def eval_keds_merged(merged_res_dir_path):
    pass

def get_candidate_path(method):
   candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} {config.train_or_test}_top100_sorted.json'
   return candidate_res_path

def eval_keds(indicate_run_name = 'single_run', indicate_id = 0):
    retrieve_res_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/{config.data_field}/' + indicate_run_name + '/'
    
    max_ind = get_max_ind(retrieve_res_dir_path)

    target_id = indicate_id if indicate_id != 0 else max_ind
    subdir_path = (retrieve_res_dir_path + str(target_id) + '/')
    print(subdir_path)

    for method in sparse_methods:            
        retrieve_res_path = subdir_path + 'all_org.json'

        method_res_dir_path = subdir_path + method + '/'
        os.makedirs(method_res_dir_path, exist_ok=True)

        
        candidate_res_path = get_candidate_path(method)
        
        retrieve_res = json_load(retrieve_res_path)['result']
        candidate_res = json_load(candidate_res_path)
        
        rerank_res = gen_rerank_ids(retrieve_res, candidate_res, config.k)
        eval_res = simple_eval(rerank_res, config.k)

        rerank_path = method_res_dir_path + f'{method}_rerank.json'
        json_dump(rerank_res, rerank_path)

        eval_path = rerank_path.replace('.json', f'_eval.json')
        json_dump(eval_res, eval_path)

        # simple_eval(candidate_res_path, candidate_res_path.replace('.json', '_eval.json'))


def get_best_ndcg5():
    from util import json_load
    # assert config.test_collection_name == 'ntcir', "only for ntcir, for ACORDAR, you should use 'merge_folds_results method'"
    
    multi_run_name = 'multi_run/37'
    multi_run_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/metadata/' + multi_run_name
    
    for method in sparse_methods:
        max_ndcg5 = 0
        max_ind = 0
        for dir_name in sorted(os.listdir(multi_run_dir_path)):
            eval_res_path = f'{multi_run_dir_path}/{dir_name}/{method}/{method}_rerank_eval.json'
            mean_metrics = json_load(eval_res_path)['mean_all']
            print(dir_name, mean_metrics['ndcg_cut_5_mean'])
            if mean_metrics['ndcg_cut_5_mean'] > max_ndcg5:
                max_ndcg5 = mean_metrics['ndcg_cut_5_mean']
                max_ind = dir_name
        print('max ndcg5: ', max_ndcg5, 'max ind: ', max_ind)

def print_ndcg_mutli_run():
    multi_run_name = 'multi_run/37'
    multi_run_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/metadata/' + multi_run_name
    for method in sparse_methods:
        for dir_name in sorted(os.listdir(multi_run_dir_path)):
            res_path = f'{multi_run_dir_path}/{dir_name}/all_org.json'
            res_config = json_load(res_path)['retrieve_config']

            eval_res_path = f'{multi_run_dir_path}/{dir_name}/{method}/{method}_rerank_eval.json'
            mean_metrics = json_load(eval_res_path)['mean_all']
            ndcg5 = mean_metrics['ndcg_cut_5_mean']
            ndcg10 = mean_metrics['ndcg_cut_10_mean']
            print('ent_word_sim_op', res_config['ent_word_sim_op'], 'word_sim_op', res_config['word_sim_op'], 'qd_ent_score_op', res_config['qd_ent_score_op'], 'qd_word_score_op', res_config['qd_word_score_op'], 'ndcg5', ndcg5, 'ndcg10', ndcg10)

def print_with_sparse_ndcg():
    from merge_score import get_path_list
    for (res1path, res2path, merge_path, op) in get_path_list():
        merge_eval_path = merge_path.replace('.json', '_eval.json')
        mean_metrics = json_load(merge_eval_path)['mean_all']
        ndcg5 = mean_metrics['ndcg_cut_5_mean']
        ndcg10 = mean_metrics['ndcg_cut_10_mean']
        
        # drop tail file name
        all_org_path = '/' + '/'.join(res1path.split('/')[:-2]) + '/all_org.json'

        sparse_method_name = res1path.split('/')[-2]

        res1_config = json_load(all_org_path)['retrieve_config']

        # print formatted, each has 10 width
        print(f'{sparse_method_name:10}', 'ent_word_sim_op', res1_config['ent_word_sim_op'], 'word_sim_op', res1_config['word_sim_op'], 'qd_ent_score_op', res1_config['qd_ent_score_op'], 'qd_word_score_op', res1_config['qd_word_score_op'], 'ndcg5', ndcg5, 'ndcg10', ndcg10)




def get_eval_test_ndcg(eval_res, fold_id):
    from util import get_5split_query
    queries_5split = get_5split_query()
    test_split_id = (fold_id + 4) % 5

    eval_ndcgs = []
    test_ndcgs = []

    test_q_num = 0
    for qid, metrics in eval_res['each query'].items():
        if int(qid) in queries_5split[test_split_id]:
            test_ndcgs.append(metrics['ndcg_cut_5'])
            test_q_num += 1
        else:
            eval_ndcgs.append(metrics['ndcg_cut_5'])

    return np.mean(eval_ndcgs), np.mean(test_ndcgs), test_q_num

def merge_folds_result(save_ind = ''):
    if save_ind != '':
        if not os.path.exists(f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/{config.data_field}/{config.res_dir_name}/{save_ind}/'):
            os.mkdir(f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/{config.data_field}/{config.res_dir_name}/{save_ind}/')
    # multi_run_id = 'multi_run_matrix2'

    retrieve_res_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/{config.data_field}/'
    # if multi_run_id != '': retrieve_res_dir_path += multi_run_id + '/'
    
    for method in sparse_methods:
        start_ind = 1
        end_ind = 121

        best_eval_ndcg_each_fold = [0 ,0, 0, 0, 0]
        best_eval_ndcg_each_fold_ind = [0 ,0, 0, 0, 0]
        test_ndcg_each_fold = [0 ,0, 0, 0, 0]

        query_test_num_splits = [0 ,0, 0, 0, 0]

        for i in range(start_ind, end_ind + 1):
            res_dir_path = retrieve_res_dir_path + f'{config.res_dir_name}/{i}/'
            res_path = res_dir_path + method + f'_org.json'
            if not os.path.exists(res_path): 
                assert 0, res_path
            candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} top100_sorted.json'
            eval(res_path, candidate_res_path, config.k)
            eval_res_path = res_path.replace('_org.json', '_rerank_eval.json')
            eval_res = json_load(eval_res_path)

            for fold_id in range(0, 5):
                eval_ndcg, test_ndcg, test_q_num = get_eval_test_ndcg(eval_res, fold_id)
                query_test_num_splits[fold_id] = test_q_num
                if eval_ndcg > best_eval_ndcg_each_fold[fold_id]:
                    best_eval_ndcg_each_fold[fold_id] = eval_ndcg
                    best_eval_ndcg_each_fold_ind[fold_id] = i
                    test_ndcg_each_fold[fold_id] = test_ndcg
        
        """
            micro avg
        """
        from util import get_5split_query
        queries_5split = get_5split_query()
    
        
        test_ndcg_total = 0
        for i in range(5):
            test_ndcg_total += test_ndcg_each_fold[i] * query_test_num_splits[i]
        
        print(sum(query_test_num_splits))
        test_ndcg_mean = test_ndcg_total / sum(query_test_num_splits)        

        print(f'{method} best eval ndcg: {best_eval_ndcg_each_fold}, test ndcg: {test_ndcg_each_fold}', 'best ind: ', best_eval_ndcg_each_fold_ind)
        print(f'{method} test avg: {test_ndcg_mean}')
        
        merged_res = {
            'best_eval_ndcg_each_fold': best_eval_ndcg_each_fold,
            'best_eval_ndcg_each_fold_ind': best_eval_ndcg_each_fold_ind,
            'test_ndcg_each_fold': test_ndcg_each_fold,
            'test_ndcg_mean': test_ndcg_mean,
            'method': method,
            'res_dir_name': config.res_dir_name,
            'result': {}
        }
        for fold_id in range(0, 5):
            best_ind = best_eval_ndcg_each_fold_ind[fold_id]
            best_ind_res = json_load(retrieve_res_dir_path + f'{config.res_dir_name}/{best_ind}/{method}_org.json')['result']
            best_ind_fold_res = {}
            for qid, res_one_q in best_ind_res.items():
                if int(qid) in queries_5split[(fold_id + 4) % 5]:
                    best_ind_fold_res[qid] = res_one_q
            merged_res['result'].update(best_ind_fold_res)
        
        json_dump(merged_res, retrieve_res_dir_path + f'{config.res_dir_name}/{save_ind}/{method}_{save_ind}_org.json')



def eval_multi_run():
    suffix = 'multi_run/37/'
    multi_run_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/rerank/{config.embedding_method}/{config.data_field}/{suffix}'
    for fname in os.listdir(multi_run_dir_path):        
        eval_keds(suffix, int(fname))

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


def eval_one_monobert_res():
    res_path = '/home/yourname/code/erm/data/retrieve_results/monobert_results/test_linear_combs_ntcir/3/BM25 [m]_comb_org.json' # '/home/yourname/code/erm/data/retrieve_results/monobert_results/monoBERT_res_ntcir_train_org.json'
    rerank_path = res_path.replace('_org.json', '_rerank.json')
    eval_path = rerank_path.replace('_rerank.json', '_rerank_eval.json')

    method = 'BM25 [m]'
    if config.test_collection_name == 'ntcir':
                candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} {config.train_or_test}_top100_sorted.json'
    else: candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} top100_sorted.json'

    res = json_load(res_path)['result']
    # for q, v in res.items():
    #     res[q] = [(int(did), score) for did, score in v.items()]


    candidate_res = json_load(candidate_res_path)
    rerank_res, eval_res = eval(res, candidate_res, config.k)

    json_dump(rerank_res, rerank_path)
    json_dump(eval_res, eval_path)

def eval_monobert_combs():
    from configs import sparse_methods
    res_dir_path = '/home/yourname/code/erm/data/retrieve_results/monobert_results/multi_run_linear_combs_ntcir_train/'
    for subdir in os.listdir(res_dir_path):
        subdir_path = res_dir_path + subdir + '/'
        for method in sparse_methods:
            if config.test_collection_name == 'ntcir':
                candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} {config.train_or_test}_top100_sorted.json'
            else: candidate_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} top100_sorted.json'

            candidate_res = json_load(candidate_res_path)

            retrieve_res_path = subdir_path + f'{method}_comb_org.json'
            # '/home/yourname/code/erm/data/retrieve_results/monobert_results/multi_run_linear_combs_ntcir/2/BM25 [m]_org.json' 
            # # f'/home/yourname/code/erm/data/retrieve_results/monobert_results/each_sparse_results/{method} monoBERT_res_list_ACORDAR_org.json' # f'/home/yourname/code/erm/data/retrieve_results/monobert_results/multi_run_linear_combs/2/{method}_org.json'

            retrieve_res = json_load(retrieve_res_path)['result']

            rerank_res, eval_res = eval(retrieve_res, candidate_res, config.k)

            rerank_path = retrieve_res_path.replace('_org.json', f'_rerank.json')
            json_dump(rerank_res, rerank_path)
            
            eval_res_path = rerank_path.replace('_rerank.json', '_rerank_eval.json')
            json_dump(eval_res, eval_res_path)

def eval_dpr_res():
    dpr_rerank_dir_path = '/home/yourname/code/erm/data/retrieve_results/condenser/from_cocondenser_marco/ACORDAR/'
    # '/home/yourname/code/erm/data/retrieve_results/condenser/ft_nq_pt/' # '/home/yourname/code/erm/data/retrieve_results/condenser/new_ntcir'
    for fname in os.listdir(dpr_rerank_dir_path):
        if not fname.endswith('rerank.json'): continue
        rerank_res_path = dpr_rerank_dir_path + '/' + fname
        rerank_res = json_load(rerank_res_path)
        eval_res = simple_eval(rerank_res, config.k)
        eval_res_path = rerank_res_path.replace('_rerank.json', '_rerank_eval.json')
        json_dump(eval_res, eval_res_path)

def eval_bert_reranker_result():
    # find path of all /home/yourname/code/erm/data/retrieve_results/bert_reranker/results/**/**/test.json
    bert_reranker_dir_path = '/home/yourname/code/erm/data/retrieve_results/bert_reranker/results_finetune/' # '/home/yourname/code/erm/data/retrieve_results/bert_reranker/results/'
    for subdir in os.listdir(bert_reranker_dir_path):
        from_to = subdir.split('_')[1]
        if from_to == 'acordar2acordar': continue
        if '2' not in from_to: continue
        res_test_collection_name = from_to.split('2')[1]
        if res_test_collection_name not in config.test_collection_name: continue
        subdir_path = bert_reranker_dir_path + subdir + '/'
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = subdir_path + subsubdir + '/'
            for fname in os.listdir(subsubdir_path):
                if not fname.endswith('test.json'): continue
                test_res_path = subsubdir_path + fname
                print('test_res_path', test_res_path)
                test_res = json_load(test_res_path)
                eval_res = simple_eval(test_res, config.k)
                eval_res_path = test_res_path.replace('test.json', 'test_eval.json')
                json_dump(eval_res, eval_res_path)


def tmp_eval():
    res_dir_path = '/home/yourname/code/erm/data/retrieve_results/bert_reranker/simple_arrange/pt_ft/ACORDAR' # '/home/yourname/code/erm/data/retrieve_results/ACORDAR/rerank/{config.embedding_method}/metadata/single_run/1124/BM25 [m]/with_sparse_merged/BM25 [m]/with_ir_sota_merged'

    for fname in os.listdir(res_dir_path):
        if fname.endswith('eval.json') or fname.endswith('merged.json'): continue
        res_path = res_dir_path + '/' + fname
        res = json_load(res_path) # ['result']
        eval_res = simple_eval(res, config.k)
        eval_res_path = res_path.replace('.json', '_eval.json')
        json_dump(eval_res, eval_res_path)

def eval_merge(path_list = None):
    from merge_score import get_path_list
    if path_list is None:
        path_list = get_path_list()
    for (res1_path, res2_path, merge_path, op) in path_list:
        merge_eval_path = merge_path.replace('.json', '_eval.json')
        # if os.path.exists(merge_eval_path): continue
        merged_res = json_load(merge_path)['result']
        eval_res = simple_eval(merged_res, config.k)
        json_dump(eval_res, merge_eval_path)


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

        # eval_merged(sys.argv[1].split('_')[-1], int(sys.argv[2]), 'single')

        # for multi_run 4 op
        # for i in range(1, 91): # [1, 2, 4, 5, 6]:
        #     eval_merged(sys.argv[1].split('_')[-1], i)
