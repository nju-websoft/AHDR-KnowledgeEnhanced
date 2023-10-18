import json
import signal
import os
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def json_load(path):
    if not os.path.isfile(path):
        print(path, 'invalid path')
        return None
    with open(path, 'r') as fp:
        return json.load(fp)
    
def json_dump(data, path):
    with open(path, 'w+') as fp:
        json.dump(data, fp, indent=2)


def get_max_ind(dir_path):
    names = os.listdir(dir_path)
    # check name is an int number
    num_names = list(filter(lambda x: x.isdigit(), names))
    return max([int(x) for x in num_names]) if num_names else 0


def get_metadata_dict():
    from configs import common_config
    import csv
    if common_config.test_collection_name == 'ACORDAR':
        metadata_dict_path = '/home/yourname/code/Scripts-for-DPR/file/ctxs_for_embedding/metadata.tsv'
        m_str_dict = {}
        with open(metadata_dict_path, 'r') as fp1:
            rd1 = csv.reader(fp1, delimiter='\t')
            for row in rd1:
                if row[0] == 'id': continue
                text = row[1].replace('[CLS] ','').replace(' [SEP]','')
                m_str_dict[int(row[0])] = text
        return m_str_dict
    else:
        from database import ntcir_cursor_org
        ntcir_cursor_org.execute('SELECT dataset_id, metadata from ntcir_metadata_info;')
        # print('ntcir_metadata not implemented yet')
        m_str_dict = {}
        for (dataset_id, metadata) in ntcir_cursor_org.fetchall():
            m_str_dict[dataset_id] = metadata
        return m_str_dict

def get_5split_query():
    query_5split_dir_path = '/home/yourname/code/Scripts-for-DPR/file/query_for_validation/for_ACORDAR/'
    queries_split = {}
    name2id = json_load('/home/yourname/code/erm/data/topic_link_results/name2id.json')
    binary_train_data_dir_path = '/home/yourname/code/erm/data/binary_train_data/'

    for split in range(0, 5):
        query_path = query_5split_dir_path + 'query_split' + str(split) + '.tsv'
        queries_split[split] = set()
        with open(query_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                query = line.split('\t')[0]
                qid = name2id[query]

                queries_split[split].add(int(qid))

    return queries_split


def get_candidates(method_name, k):
    """
    ret: dict, key: query_id, value: list of (doc_id, score)
    """
    from configs import sparse_methods, common_config
    candidate_dir_path = f'../data/retrieve_results/{common_config.test_collection_name}/candidates/'
    target_name = list(filter(lambda x: method_name in x and x.endswith('sorted.json'), os.listdir(candidate_dir_path)))
    assert len(target_name) == 1
    target_name = target_name[0]
    candidate_path = candidate_dir_path + target_name
    return json_load(candidate_path)[:k]


def get_topk_candidate_set(k):
    from configs import common_config as config, sparse_methods

    dids = set()
    for method_name in sparse_methods:
        sparse_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method_name} test_top100_sorted.json'
        sparse_res = json_load(sparse_res_path)

        for qid, res in sparse_res.items():
            one_topk = [did for (did, score) in res[:k]]
            for did in one_topk:
                dids.add(did)

    return dids


def merge_two_score(score1, score2, op):
    """
    score_op notes:
    1: 2-stage avg: (sum1/len1 + sum2/len2) / 2
    2: harmonic avg: 2 * sum1 * sum2 / (sum1 + sum2)
    3: total avg: sum1 + sum2 / len1 + len2
    4: Geometric avg: sqrt(sum1 * sum2)
    5: max: max(sum1, sum2)
    6: min: min(sum1, sum2)
    """
    if op == 1:
        return (score1 + score2) / 2
    elif op == 2:
        return 2 * score1 * score2 / (score1 + score2) if score1 + score2 != 0 else 0
    elif op == 3:
        return score1 + score2
    elif op == 4:
        if score1 < 0 or score2 < 0:
            # print('score1', score1, 'score2', score2)
            # assert 0, f'wrong score1 {score1} or score2 {score2}'
            return 0
        return math.sqrt(score1 * score2)
    elif op == 5:
        return max(score1, score2)
    elif op == 6:
        return min(score1, score2)
    else:
        print('wrong op', op)
        raise NotImplementedError