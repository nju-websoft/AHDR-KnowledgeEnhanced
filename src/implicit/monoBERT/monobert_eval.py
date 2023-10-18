import numpy as np
import pytrec_eval
import json

# def json_load(path):
#     with open(path, 'r') as f:
#         return json.load(f)
#
# qrels = json_load('data/qrels.json')

def simple_eval(qrels, res_dict, k):
    """
    res_dict format:
    key: query_id
    value: list-type dataset_ids
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map_cut.5','map_cut.10','ndcg_cut.5','ndcg_cut.10'})
    run = dict([(query_id, dict((str(did), score) for did, score in did_scores[:k])) for (query_id, did_scores) in res_dict.items()])
    rank_metric_val = evaluator.evaluate(run)
    mean_dict = {}
    for metric_name in ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_5', 'map_cut_10']:
        mean_dict[metric_name+'_mean'] = np.around(np.mean([val[metric_name] for val in rank_metric_val.values()]), 4)

    output_res = {
        'each query': rank_metric_val,
        'mean': mean_dict,
    }

    return output_res
