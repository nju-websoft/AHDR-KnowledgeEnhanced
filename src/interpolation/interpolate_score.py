import os
import itertools
from util import json_load, json_dump, get_qrels
from configs import eval_config as config
from eval import simple_eval

test_collection_name = config.test_collection_name

def normalize_one_res(did_score_list, normalize_way = 'max-min'):
    if len(did_score_list) == 0:
        return []
    # normalize by max-min
    max_value = max([score for _, score in did_score_list])
    min_value = min([score for _, score in did_score_list])
    if max_value == min_value:
        return [(did, 1) for did, _ in did_score_list]
    else:
        return [(did, (score - min_value) / (max_value - min_value)) for did, score in did_score_list]

def interpolate2res(res1, res2, interpolate_weight, k = 10):
    interpolate_res = {}
    for qid in res2.keys():
        # print(qid)
        
        res2_one = res2[qid][:k]
        normalized_res2_one = normalize_one_res(res2_one)
        res2_one_dict = dict(normalized_res2_one)
        if qid not in res1.keys() or res1[qid] is None: 
            print('qid not in res1.keys() or res1[qid] is None', qid)
            res1[qid] = res2_one

        # be careful, only select those in sparse res
        res1_one = res1[qid][:k]
        normalized_res1_one = normalize_one_res(res1_one)        
        res1_one_dict = dict(normalized_res1_one)
                    
        if set(res1_one_dict.keys()) != set(res2_one_dict.keys()):
            print(f'qid: {qid}, res1: {res1_one_dict.keys()}, res2: {res2_one_dict.keys()}')

        interpolate_res_one_dict = {}        
        for did in res2_one_dict.keys():
            interpolate_res_one_dict[did] = interpolate_weight * res1_one_dict.get(did, 0) + (1 - interpolate_weight) * res2_one_dict[did]

        interpolate_res[qid] = list(interpolate_res_one_dict.items())
    return interpolate_res

def get_interpolate_eval_list(res1, res2, ground_truth_qrels = None):
    # set res1 res2 be the form of str -> [(str, float)]
    res1 = {str(qid): [(str(did), score) for did, score in res] for qid, res in res1.items()}
    res2 = {str(qid): [(str(did), score) for did, score in res] for qid, res in res2.items()}
    
    if ground_truth_qrels is None:
        ground_truth_qrels = qrels
    interpolate_eval_list = []
    candidate_interpolate_weights = [0.1 * i for i in range(11)]
    for interpolate_weight in candidate_interpolate_weights:
        interpolated_res = interpolate2res(res1, res2, interpolate_weight)
        
        eval_res = simple_eval(interpolated_res, 10, ground_truth_qrels)
        # delete key 'eval_config' in dict
        eval_res.pop('eval_config')
        interpolate_eval_list.append((eval_res, interpolate_weight)) 

    return interpolate_eval_list


def get_3res_interpolate(res1, res2, res3, ground_truth_qrels = None, k = 10):
    if ground_truth_qrels is None:
        ground_truth_qrels = qrels
    res1 = {str(qid): [(str(did), score) for did, score in res] for qid, res in res1.items()}
    res2 = {str(qid): [(str(did), score) for did, score in res] for qid, res in res2.items()}
    res3 = {str(qid): [(str(did), score) for did, score in res] for qid, res in res3.items()}

    interpolate_res = {}
    for qid in res2.keys():
        
        res2_one = res2[qid][:k]
        normalized_res2_one = normalize_one_res(res2_one)
        res2_one_dict = dict(normalized_res2_one)
        if qid not in res1.keys() or res1[qid] is None: 
            print('qid not in res1.keys() or res1[qid] is None', qid)
            res1[qid] = res2_one

        res1_one = res1[qid][:k]
        normalized_res1_one = normalize_one_res(res1_one)        
        res1_one_dict = dict(normalized_res1_one)

        res3_one = res3[qid][:k]
        normalized_res3_one = normalize_one_res(res3_one)
        res3_one_dict = dict(normalized_res3_one)

        if set(res1_one_dict.keys()) != set(res2_one_dict.keys()) or set(res1_one_dict.keys()) != set(res3_one_dict.keys()):
            print(f'qid: {qid}, res1: {res1_one_dict.keys()}, res2: {res2_one_dict.keys()}', f'res3: {res3_one_dict.keys()}')

        interpolate_res_one_dict = {}        
        for did in res2_one_dict.keys():
            interpolate_res_one_dict[did] = (res1_one_dict.get(did, 0) + res2_one_dict.get(did, 0) + res3_one_dict.get(did, 0)) / 3

        interpolate_res[qid] = list(interpolate_res_one_dict.items())        

    eval_res = simple_eval(interpolate_res, 10, ground_truth_qrels)
    # delete key 'eval_config' in dict
    eval_res.pop('eval_config')

    return eval_res

def interpolate():
    # flag = False
    interpolate_type = 'ie' # is, es, ie, ies
    assert interpolate_type in ['ib', 'eb', 'ie', 'ieb'], f'interpolate_type: {interpolate_type}'
    candidate_finetune_status = ['w_finetune', 'wo_finetune'] if interpolate_type.startswith('i') else ['']
    candidate_irsotas = ['ANCE', 'ColBERT', 'condenser', 'monoBERT', 'monoT5'] if interpolate_type.startswith('i') else ['']
    for test_collection_name, finetune_status, irsota in itertools.product(['ACORDAR', 'NTCIR-E'], candidate_finetune_status, candidate_irsotas):
        global qrels
        qrels = get_qrels(test_collection_name)
        config.test_collection_name = test_collection_name
        print(test_collection_name, finetune_status, irsota)
        
        if interpolate_type.startswith('i'):
            implicit_res_dir_path = f'/home/yourname/code/erm/data/retrieve_results/irsota/{irsota}/{finetune_status}/{test_collection_name}/'
            # list fname in dir and get implicit eval res ends with eval.json
            implicit_res_eval_path_list = [os.path.join(implicit_res_dir_path, fname) for fname in os.listdir(implicit_res_dir_path) if fname.endswith('eval.json')]
            assert len(implicit_res_eval_path_list) == 1
            implicit_res_eval_path = implicit_res_eval_path_list[0]
            implicit_res_path = implicit_res_eval_path.replace('_eval.json', '.json')            
            ikedes_res = json_load(implicit_res_path)
        else:
            ikedes_res = None

        explicit_res_path = f'/home/yourname/code/erm/data/retrieve_results/explicit/{test_collection_name}/14/BM25 [m]/BM25 [m]_rerank.json'
        explicit_res = json_load(explicit_res_path)

        bm25_res_path = f'/home/yourname/code/erm/data/retrieve_results/{test_collection_name}/candidates/BM25 [m] test_top100_sorted.json'

        bm25_res = json_load(bm25_res_path)

        interpolate_candidates = {
            'explicit': explicit_res,
            'implicit': ikedes_res,
            'bm25': bm25_res
        }

        key3 = None
        if interpolate_type == 'ib':
            key1 = 'implicit'
            key2 = 'bm25'
            interpolate_name_prefix = f'{irsota}_{finetune_status}_bm25'
        elif interpolate_type == 'eb':
            key1 = 'explicit'
            key2 = 'bm25'
            interpolate_name_prefix = f'explicit_bm25'
        elif interpolate_type == 'ie':
            key1 = 'implicit'
            key2 = 'explicit'
            interpolate_name_prefix = f'{irsota}_{finetune_status}_explicit'
        elif interpolate_type == 'ieb':
            key1 = 'implicit'
            key2 = 'explicit'
            key3 = 'bm25'
            interpolate_name_prefix = f'{irsota}_{finetune_status}_explicit_bm25'

        res1 = interpolate_candidates[key1]
        res2 = interpolate_candidates[key2]
    
        res3 = None if key3 is None else interpolate_candidates[key3]

        if interpolate_type != 'ieb':
            interpolate_eval_list = get_interpolate_eval_list(res1, res2)
            interpolate_eval_list_dump_path = f'/home/yourname/code/erm/data/retrieve_results/interpolate_results/{test_collection_name}/{interpolate_type}_interpolated/{interpolate_name_prefix}_score_list.json'
            json_dump(interpolate_eval_list, interpolate_eval_list_dump_path)

        if res3 is not None:
            dump_res = get_3res_interpolate(res1, res2, res3)
            dump_res_path = f'/home/yourname/code/erm/data/retrieve_results/interpolate_results/{test_collection_name}/{interpolate_type}_interpolated/{interpolate_name_prefix}_eval.json'
            json_dump(dump_res, dump_res_path)

if __name__ == '__main__':
    interpolate()
