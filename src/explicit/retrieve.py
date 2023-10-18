import os
import numpy as np
from gensim.models import Word2Vec
from database import ntcir_cursor_org
from util import json_load, json_dump, get_max_ind, merge_two_score
from multiprocessing import Pool, Manager
from configs import retrieve_dict_config, retrieve_config as config, topic_link_res_path, sparse_methods
from logger import log

worker_num = 40
rdf_vectors = None
wiki2vec = None
pair2score = {}
def load_embedding_model():
    from wikipedia2vec import Wikipedia2Vec
    global wiki2vec
    wiki2vec_pkl_path = '/home/yourname/code/erm/data/wiki2vec/100d/enwiki_20180420_100d.pkl' if config.dim == 100 else \
        '/home/yourname/code/erm/data/wiki2vec/500d/enwiki_20180420_500d.pkl'
    wiki2vec = Wikipedia2Vec.load(wiki2vec_pkl_path)
    if config.embedding_method == 'wiki2vec': 
        log.info('load model done')
        return

    global rdf_vectors
    model_path = None
    if config.embedding_method == 'kgtk':
        init_pair2score()
    else:
        if config.embedding_method == 'dbpedia_rdf2vec_sg_500': model_path = '/home/yourname/code/erm/data/rdf2vec/dbpedia/DB2Vec_sg_500_5_5_15_4_500'
        elif config.embedding_method == 'dbpedia_rdf2vec_sg_200': model_path = "/home/yourname/code/erm/data/rdf2vec/dbpedia/dbpedia_500_4_sg_100"
        elif config.embedding_method == 'wikidata_rdf2vec_sg_200_1': model_path = "/home/yourname/code/erm/data/rdf2vec/wikidata/wikid2Vec_sg_200_5_5_15_4_500"
        elif config.embedding_method == 'wikidata_rdf2vec_sg_200_2': model_path = "/home/yourname/code/erm/data/rdf2vec/wikidata/wikid2Vec_sg_200_7_4_15_4_500"
        else:
            print('illeagle config.embedding_method', config.embedding_method)
            exit(0)
        model = Word2Vec.load(model_path)
        rdf_vectors = model.wv
    log.info('load model done')

def init_pair2score():
    ntcir_cursor_org.execute(f'SELECT query_entity_uri, dataset_entity_uri, text_similarity FROM `query_dataset_entity_similartity_kgtk`')
    for (query_entity_uri, dataset_entity_uri, text_similarity) in ntcir_cursor_org.fetchall():
        if query_entity_uri not in pair2score.keys(): pair2score[query_entity_uri] = {}
        pair2score[query_entity_uri][dataset_entity_uri] = text_similarity
    log.info(f'all pair2score len: {len(pair2score)}')


def get_entity_vector(title, do_normalize = True):
    if title is None or title == '': return None
    title = ' '.join(title.split('_'))
    if config.embedding_method == 'wiki2vec':
        try:
            v1 = wiki2vec.get_entity_vector(title)
        except Exception:
            return None
    elif config.embedding_method.startswith('dbpedia'):
        text = 'dbr:'+'_'.join(title.split(' '))
        
        if text not in rdf_vectors.vocab: return None
        v1 = rdf_vectors[text]
    elif config.embedding_method.startswith('wikidata'):
        text = '_'.join(title.split(' '))
        if text not in rdf_vectors.vocab: return None
        v1 = rdf_vectors[text]
    return v1 / np.linalg.norm(v1) if do_normalize else v1

def get_word_vector(title, do_normalize = True):
    try:
        v1 = wiki2vec.get_entity_vector(title)
    except Exception:
        # print('wiki2vec get entity vector error', title)
        try:
            title = title.lower()
            v1 = wiki2vec.get_word_vector(title)
        except Exception as e: return None
    return v1 / np.linalg.norm(v1) if do_normalize else v1

def get_all_dids():
    # drop suffix number
    test_collection_name = 'ntcir' if config.test_collection_name.startswith('ntcir') else config.test_collection_name
    ntcir_cursor_org.execute(f'select distinct(dataset_id) from {test_collection_name}_metadata_info')
    dids = [did for (did, ) in ntcir_cursor_org.fetchall()]
    return dids

def get_one_did_entity_info(did):
    test_collection_name = 'ntcir' if config.test_collection_name.startswith('ntcir') else config.test_collection_name
    term_table_name = f'{test_collection_name}_{config.data_field}_link_result_{config.link_method}'

    confidence_name = 'MD_confidence' if config.link_method == 'REL' else 'rho' # 'link_probability' # 'rho'
    # rho_suffix = 'and rho >= 0.1' if config.filter_rho else ''
    ntcir_cursor_org.execute(f'select wiki_title, {confidence_name} from {term_table_name} where dataset_id = {did}')
    return list(ntcir_cursor_org.fetchall())

def init_one_did_vectors(did, m_d_vectors_total):
    global dataset_entity_info

    ew_vectors = build_entity_and_word_vector_from_entity_info(dataset_entity_info.get(did, []))
    if ew_vectors is None:
        return

    m_d_vectors_total[did] = ew_vectors
    sz = len(m_d_vectors_total)

d_indexes = {}
d_vectors_total = {}
d_confidences = {}
dataset_entity_info = {}
dids = get_all_dids()

query_entity_info = {}
q_indexes = {}
q_vectors_total = {}
q_confidences = {}
q_avg_cosine_similarities = {}

def build_entity_and_word_vector_from_entity_info(infos):
    vectors = []
    for ind, (spot, wiki_title, confidence) in enumerate(infos):
        if config.embedding_method == 'kgtk':
            evt = wiki_title
        else:
            evt = get_entity_vector(wiki_title, False)

        word_vectors = []
        for word in spot.split(' '):
            word = word.lower()
            wv = get_word_vector(word, False)
            if wv is None: continue
            word_vectors.append(wv)
        
        if evt is not None: # or len(word_vectors) > 0:
            if config.use_word_embedding and config.word_embedding_use_way == 'one' and config.embedding_method != 'kgtk':
                # find the most similar word vector to entity vector by cosine similarity
                if len(word_vectors) > 0:
                    cos_similarities = [cos_sim(evt, wv) for wv in word_vectors]
                    max_ind = 0
                    for i in range(1, len(cos_similarities)):
                        if cos_similarities[i] > cos_similarities[max_ind]:
                            max_ind = i
                    wvt = word_vectors[max_ind]
                    word_vectors = [wvt]
            vectors.append((evt, word_vectors))
    if vectors == []: return None
    else: return vectors

def init_entity_and_word_vectors():
    global query_entity_info, dataset_entity_info
    global q_vectors_total, d_vectors_total
    # global d_confidences

    link_method = config.link_method
    # rank = 1 # only use the top 1 entity

    # init query vectors
    if link_method == "TAGME":
        rho_lb = config.filter_rho_lb
        rho_ub = config.filter_rho_ub
        link_pb_lb = config.filter_link_pb_lb
        link_pb_ub = config.filter_link_pb_ub

        table_name = f'{config.test_collection_name}_query_link_result_{config.link_method}'
        confidence_name = 'MD_confidence' if config.link_method == 'REL' else 'rho'
        sql_where = f' where rho >= {rho_lb} and rho <= {rho_ub} and link_probability >= {link_pb_lb} and link_probability <= {link_pb_ub}' if config.do_entity_filter else ''
        sql = f'SELECT query_id, wiki_title, {confidence_name}, spot from {table_name} {sql_where} where train_or_test like "{config.train_or_test}"'
        ntcir_cursor_org.execute(sql)
        log.info('load query_entity_info')
        for (qid, wiki_title, confidence, spot) in ntcir_cursor_org.fetchall():
            if qid not in query_entity_info.keys(): query_entity_info[qid] = []
            query_entity_info[qid].append((spot, wiki_title, confidence))
            
    elif link_method=="FALCON2":
        table_name = f'{config.test_collection_name}_query_link_result_{config.link_method}'
        sql = f'SELECT query_id, surface_form, URI, rank FROM {table_name} where rank = 1'
        # load entity info
        ntcir_cursor_org.execute(sql)
        log.info('load query_entity_info')
        for (qid, spot, wiki_title, rank) in ntcir_cursor_org.fetchall():
            # log.info(f'{qid} {spot} {wiki_title} {rank}')
            if qid not in query_entity_info.keys(): query_entity_info[qid] = []
            confidence = 1
            query_entity_info[qid].append((spot, wiki_title, confidence))
    elif link_method=="REL":
        table_name = f'{config.test_collection_name}_query_link_result_{config.link_method}'
        sql = f'SELECT query_id, spot, wiki_title FROM {table_name}'
        # load entity info
        ntcir_cursor_org.execute(sql)
        log.info('load query_entity_info')
        for (qid, spot, wiki_title) in ntcir_cursor_org.fetchall():
            # log.info(f'{qid} {spot} {wiki_title} {rank}')
            if qid not in query_entity_info.keys(): query_entity_info[qid] = []
            confidence = 1
            query_entity_info[qid].append((spot, wiki_title, confidence))
    
    log.info('load query_entity_info done')

    for qid, infos in query_entity_info.items():
        ew_vectors = build_entity_and_word_vector_from_entity_info(infos)
        if ew_vectors is None:
            # print(qid, 'available q entity num is 0!')
            continue
        q_vectors_total[qid] = ew_vectors

    # init dataset vectors
    log.info('load dataset_entity_info')
    data_field = 'metadata' # config.data_field
    test_collection_name = 'ntcir' if config.test_collection_name.startswith('ntcir') else config.test_collection_name
    table_name = f'{test_collection_name}_{data_field}_link_result_{config.link_method}'
    if config.link_method == 'TAGME':
        sql = f'SELECT dataset_id, wiki_title, {confidence_name}, spot from {table_name} {sql_where}'
        ntcir_cursor_org.execute(sql)
        for (did, wiki_title, confidence, spot) in ntcir_cursor_org.fetchall():
            if did not in dataset_entity_info.keys(): dataset_entity_info[did] = []
            dataset_entity_info[did].append((spot, wiki_title, confidence))
        log.info('load dataset_entity_info done')
    elif config.link_method == 'FALCON2':
        sql = f'SELECT dataset_id, surface_form, URI, rank FROM {table_name} where rank = 1'
        
        ntcir_cursor_org.execute(sql)
        for (did, spot, wiki_title, rank) in ntcir_cursor_org.fetchall():
            if did not in dataset_entity_info.keys(): dataset_entity_info[did] = []
            confidence = 1
            dataset_entity_info[did].append((spot, wiki_title, confidence))
        log.info('load dataset_entity_info done')
    elif link_method=="REL":
        table_name = f'{config.test_collection_name}_metadata_link_result_{config.link_method}'
        sql = f'SELECT dataset_id, spot, wiki_title FROM {table_name}'
        # load entity info
        ntcir_cursor_org.execute(sql)
        for (did, spot, wiki_title) in ntcir_cursor_org.fetchall():
            if did not in dataset_entity_info.keys(): dataset_entity_info[did] = []
            confidence = 1
            dataset_entity_info[did].append((spot, wiki_title, confidence))
    

    manager = Manager()
    m_d_vectors_total = manager.dict()

    dids = get_all_dids()
    p = Pool(worker_num)
    p.starmap(init_one_did_vectors, [(did, m_d_vectors_total) for did in dids])
    p.close()
    p.join()

    log.info('m_d len: {}'.format(len(m_d_vectors_total)))

    d_vectors_total = dict(m_d_vectors_total)
    log.info('init dataset vectors done')

def get_all_qids():
    data = json_load(topic_link_res_path)
    return list(data.keys())

def cos_sim(a, b):
    cos_score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # assert cos_score <= 1 + 1e-5 and cos_score >= 0 - 1e-5, f'cos_score is {cos_score}'
    # if cos_score < 0: cos_score = 0
    return cos_score

def wv_list_sim(a_list, b_list, op = 0):
    if len(a_list) == 0 or len(b_list) == 0: return 0
    sim_score_matrix = []
    for a in a_list:
        sim_scores = []
        for b in b_list:
            sim_scores.append(cos_sim(a, b))
        sim_score_matrix.append(sim_scores)
    if op == 0:
        # avg all sim scores
        return float(np.mean(sim_score_matrix))
    elif op == 7:
        # max of all sim scores
        return float(np.max(sim_score_matrix))
    else:
        col_max_avg = np.mean(np.max(sim_score_matrix, axis=0))
        row_max_avg = np.mean(np.max(sim_score_matrix, axis=1))
        return merge_two_score(col_max_avg, row_max_avg, op)

def matrix2score(matrix, op = 0):
    matrix = np.array(matrix)
    # get row max avg and col max avg
    row_max_avg = np.mean(np.max(matrix, axis=1))
    col_max_avg = np.mean(np.max(matrix, axis=0))

    return merge_two_score(row_max_avg, col_max_avg, op)

def rerank_one_topic_hierarchy(res_dict, qid, candidate_dids = []):
    # log.info(qid)
    global dids
    if qid not in q_vectors_total.keys():
        # print(qid, 'no annos')
        res_dict[qid] = None
        return
    
    dataset_scores = {}
    if len(candidate_dids) > 0: dids = candidate_dids

    q_vectors = q_vectors_total[qid]

    for did in dids:
        if did not in d_vectors_total.keys():
            continue
        
        qd_score_matrix = []
        qd_score_matrix_word = []

        for (q_ev, q_wv_list) in q_vectors:
            one_row_scores = []
            one_row_scores_word = []
            for (d_ev, d_wv_list) in d_vectors_total[did]:
                if config.embedding_method != 'kgtk':
                    entity_sim_score = cos_sim(q_ev, d_ev) if q_ev is not None and d_ev is not None else 0
                else:
                    q_uri = q_ev
                    d_uri = d_ev
                    if q_uri not in pair2score.keys() or d_uri not in pair2score[q_uri].keys():
                        entity_sim_score = 0
                    else:
                        entity_sim_score = float(pair2score[q_uri][d_uri])
                        if entity_sim_score < 0: entity_sim_score = 0
                word_sim_score = wv_list_sim(q_wv_list, d_wv_list, config.word_sim_op) if q_wv_list is not None and d_wv_list is not None else 0

                if config.use_entity_embedding and config.use_word_embedding and config.word_embedding_use_way == 'multi_level1':
                    entity_sim_score = merge_two_score(entity_sim_score, word_sim_score, config.ent_word_sim_op)

                one_row_scores.append(entity_sim_score)
                one_row_scores_word.append(word_sim_score)
            qd_score_matrix.append(one_row_scores)
            qd_score_matrix_word.append(one_row_scores_word)


        qd_ent_score = matrix2score(qd_score_matrix, config.qd_ent_score_op)
        qd_word_score = matrix2score(qd_score_matrix_word, config.qd_word_score_op)

        if config.use_entity_embedding and config.use_word_embedding and config.word_embedding_use_way == 'multi_level2':
            dataset_scores[did] = merge_two_score(qd_ent_score, qd_word_score, config.ent_word_sim_op) if qd_ent_score != 0 and qd_word_score !=0 else qd_ent_score + qd_word_score
        elif config.use_word_embedding and not config.use_entity_embedding:
            dataset_scores[did] = qd_word_score
        else:
            dataset_scores[did] = qd_ent_score

    rank_ids = sorted(list(dataset_scores.items()), key = lambda x: x[1], reverse = True)
    res_dict[qid] = rank_ids

def multi_rerank(candidate_did_dict):

    if config.use_tfidf_style_method:
        assert 0, 'not support'
    else:
        manager = Manager()    
        retrieve_res_dict = manager.dict()
        p = Pool(worker_num)
        p.starmap(rerank_one_topic_hierarchy, [(retrieve_res_dict, query_id, candidate_dids) for (query_id, candidate_dids) in candidate_did_dict.items()])
        p.close()
        p.join()

    return retrieve_res_dict

def rerank_start(res_dir_name = 'single_run'):
    sparse_candidate_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/'

    retrieve_res_dir_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/{config.stg}/{config.embedding_method}/{config.data_field}/{res_dir_name}/'

    if not os.path.exists(retrieve_res_dir_path):
        print('create dir', retrieve_res_dir_path)
        os.makedirs(retrieve_res_dir_path, exist_ok=True)

    max_ind = get_max_ind(retrieve_res_dir_path)
    subdir_path = retrieve_res_dir_path + f'{max_ind+1}/'
    os.makedirs(subdir_path, exist_ok=True)

    candidate_did_dict_all = {}

    for sparse_method in sparse_methods:
        log.info(sparse_method)

        candidate_file_name = f'{sparse_method} {config.train_or_test}_top100_sorted.json'
        candidate_did_scores = json_load(f'{sparse_candidate_dir_path}/{candidate_file_name}')

        for qid, did_scores in candidate_did_scores.items():
            if qid not in candidate_did_dict_all: candidate_did_dict_all[qid] = []
            candidate_did_dict_all[qid].extend([did for (did, score) in did_scores[:config.k]])

    # filter out duplicate dids
    for qid, dids in candidate_did_dict_all.items():
        candidate_did_dict_all[qid] = list(set(dids))
        
    retrieve_res = multi_rerank(candidate_did_dict_all) # rerank(f'{sparse_candidate_dir_path}/{candidate_file_name}')

    # transform retrieve_res to JSON-serializable object, float32 -> float
    retrieve_res = dict(retrieve_res)
    for qid, rank_ids in retrieve_res.items():
        if rank_ids is None: continue
        retrieve_res[qid] = [(did, float(score)) for (did, score) in rank_ids]

    retrieve_res = {
        'retrieve_config': retrieve_dict_config,
        'result': retrieve_res
    }

    json_dump(retrieve_res, subdir_path + f'all_org.json')

    # eval
    from eval import eval_keds
    eval_keds(res_dir_name, max_ind + 1)

    # merge_with_sparse
    from merge_score import get_path_and_merge

    paths_and_op_list = [] # dense_res_path, sparse_res_path, merged_res_path
    out_dir_name = 'with_sparse_merged'
    candidate_merge_op = [1]
        
    for method in sparse_methods:

        sparse_res_path = f'/home/yourname/code/erm/data/retrieve_results/{config.test_collection_name}/candidates/{method} {config.train_or_test}_top100_sorted.json'

        rerank_dir_path = subdir_path + f'{method}/'

        rerank_res_path = rerank_dir_path + f'{method}_rerank.json'

        merged_res_dir_path = rerank_dir_path + out_dir_name + '/'

        os.makedirs(merged_res_dir_path, exist_ok=True)

        for op in candidate_merge_op:
            merged_res_path = f'{merged_res_dir_path}/op_{op}_merged.json'            
            paths_and_op_list.append((rerank_res_path, sparse_res_path, merged_res_path, op))

    get_path_and_merge(paths_and_op_list)

    # eval merged
    from eval import eval_merge
    eval_merge(paths_and_op_list)

if __name__ == '__main__':
    load_embedding_model()
    init_entity_and_word_vectors()
    rerank_start()