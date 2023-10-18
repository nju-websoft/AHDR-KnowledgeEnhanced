import os
import json
from database import ntcir_cursor_org, ntcir_db
from logger import log
from util import json_load, json_dump

def sqlite3_to_mysql(link_method):
    from database import sql_list
        
    dids = sql_list('select distinct(dataset_id) from dataset_metadata')
    # for (dataset_id, ) in dids:
    #     for (spot, start, link_probability, rho, end, source_id, wiki_title) in \
    #         sql_list(f'select spot, start, link_probability, rho, end, source_id, title from metadata_entity where dataset_id = {dataset_id}'):
    #         ntcir_cursor_org.execute(f"insert ignore into metadata_link_result_TAGME (dataset_id, spot, start, link_probability, rho, end, source_id, wiki_title) values (%s,%s,%s,%s,%s,%s,%s,%s)", \
    #                     (dataset_id, spot, start, link_probability, rho, end, source_id, wiki_title))
    #     ntcir_db.commit()
    
    for (dataset_id, ) in dids:
        for (dataset_id, hash_id, metadata, finish, fail) in \
            sql_list(f'select dataset_id, hash_id, metadata, finish, fail from dataset_metadata where dataset_id = {dataset_id}'):
            ntcir_cursor_org.execute(f"insert ignore into metadata_info (dataset_id, hash_id, metadata, finish, fail) values (%s,%s,%s,%s,%s)", \
                        (dataset_id, hash_id, metadata, finish, fail))
        ntcir_db.commit()

        print(dataset_id, 'done')


def get_unstore_ids(link_method):
    ntcir_cursor_org.execute(f'SELECT DISTINCT(dataset_id) from ntcir_file_info WHERE v2_finish_TAGME = 0;')
    ids = set()
    for (did, ) in ntcir_cursor_org.fetchall():
        ids.add(did)
    return ids

def store_one_link_res(link_method, annos, did, file_id):
    for item in annos:
        if link_method == 'TAGME':
            spot = item['spot']
            start = item['start']
            link_p = item['link_probability']
            rho = item['rho']
            end = item['end']
            sid = item['id']
            title = item.get('title', None)
            # print(did, file_id, spot, start, link_p, rho, end, sid, title)
            ntcir_cursor_org.execute(f"insert ignore into {test_collection_name}_content_link_result_{link_method} (dataset_id, file_id, spot, start, link_probability, rho, end, source_id, title ) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)", \
                (did, file_id, spot, start, link_p, rho, end, sid, title))
        else:
            start = item[0]
            mention_len = item[1]
            spot = item[2]
            wiki_title = item[3]
            ED_confidence = item[4]
            MD_confidence = item[5]
            type = item[6]
            if data_field == 'content':
                ntcir_cursor_org.execute(f"insert ignore into content_link_result_{link_method} (dataset_id, file_id, spot, start, mention_length, ED_confidence, MD_confidence, type, wiki_title) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)", \
                    (did, file_id, spot, start, mention_len, ED_confidence, MD_confidence, type, wiki_title))
            else:
                ntcir_cursor_org.execute(f"insert ignore into metadata_link_result_{link_method} (dataset_id, spot, start, mention_length, ED_confidence, MD_confidence, type, wiki_title) values (%s,%s,%s,%s,%s,%s,%s,%s)", \
                    (did, spot, start, mention_len, ED_confidence, MD_confidence, type, wiki_title))

def store_link_res(link_method):
    # anno_dir_path = f'/home/yourname/code/erm/data/annos_other_fmt_{link_method}'
    anno_dir_path = '/home/yourname/code/erm/data/annos/ntcir/annos_other_fmt_TAGME' # f'/home/yourname/code/erm/data/annos/{data_field}_{link_method}'
    unstore_ids = get_unstore_ids(link_method)
    cnt = 0
    for fname in sorted(os.listdir(anno_dir_path)):
        prefix = fname.replace('.json', '')
        did = int(prefix.split('_')[0])
        fid = int(prefix.split('_')[1])
        if did not in unstore_ids: continue
        
        fpath = f'{anno_dir_path}/{fname}'
        data = json_load(fpath)
        if data_field == 'content':
            store_one_link_res(link_method, data['annotations'], did, fid)
            ntcir_cursor_org.execute(f"update {test_collection_name}_file_info set v2_finish_{link_method} = 1 where dataset_id = {did} and file_id = {fid}")
        else:
            store_one_link_res(link_method, data['annotations'], did, 0)
            ntcir_cursor_org.execute(f"update metadata_info set finish_{link_method} = 1 where dataset_id = {did}")
        ntcir_db.commit()
        cnt += 1
        if cnt % 1000 == 0: log.info(cnt)

def metadata2file():
    from database import sql_list
    dids = sql_list('select distinct(dataset_id) from dataset_metadata')
    
    metadata_dir_path = '/home/yourname/code/erm/data/ntcir_file/v2/dataset_metadata'
    for (dataset_id, ) in dids:
        for (dataset_id, hash_id, metadata, finish, fail) in \
            sql_list(f'select dataset_id, hash_id, metadata, finish, fail from dataset_metadata where dataset_id = {dataset_id}'):
            with open(f'{metadata_dir_path}/{dataset_id}.txt', 'w+') as fp:
                fp.write(metadata)

        print(dataset_id, 'done')

def store_qrels(qrels_path, type):
    with open(qrels_path, 'r') as fp1:
        qrels= json.load(fp1)
    for (topic, score_dict) in qrels.items():
        query_id = int(topic.split('-')[-1])
        for (did, score) in score_dict.items():
            ntcir_cursor_org.execute('insert into qrels (query_id, query_topic, dataset_id, score, type) \
                                     values (%s, %s, %s, %s, %s)', (query_id, topic, did, score, type))
    ntcir_db.commit()

from util import json_load, json_dump
def store_ACORDAR_link_res():
    ntcir_cursor_org.execute(f'SELECT DISTINCT(dataset_id) from ACORDAR_{data_field}_link_result_TAGME;')
    done_dids = [did for (did, ) in ntcir_cursor_org.fetchall()]
    dir_path = f'/home/yourname/code/erm/data/annos/ACORDAR/ACORDAR_{data_field}'
    cnt = 0
    for fname in os.listdir(dir_path):
        # did = int(fname.split('.')[0]) # bug !!!!
        did = int(fname.replace('.json', '').split('_')[0])
        if did in done_dids: continue
        anno_res = json_load(f'{dir_path}/{fname}')
        for item in anno_res['annotations']:
            spot = item['spot']
            start = item['start']
            link_p = item['link_probability']
            rho = item['rho']
            end = item['end']
            sid = item['id']
            title = item.get('title', None)
            # print(did, file_id, spot, start, link_p, rho, end, sid, title)
            ntcir_cursor_org.execute(f"insert ignore into ACORDAR_{data_field}_link_result_TAGME (dataset_id, spot, start, link_probability, rho, end, source_id, wiki_title ) values (%s,%s,%s,%s,%s,%s,%s,%s)", \
                (did, spot, start, link_p, rho, end, sid, title))
        ntcir_db.commit()
        cnt += 1
        if cnt % 1000 == 0: print(cnt)

def store_ntcir_file_format():
    data = json_load('/home/yourname/code/erm/src/preprocess/format_list.json')
    for (format, fid) in data:
        ntcir_cursor_org.execute('update ntcir_file_info set detect_format = %s where file_id = %s', (format, fid))
    ntcir_db.commit()

def store_ACORDAR_metadata_info():
    import csv
    metadata_path = '/home/yourname/code/Scripts-for-DPR/file/ctxs_for_embedding/metadata.tsv'
    m_data = {}
    with open(metadata_path, 'r') as fp1:
        rd1 = csv.reader(fp1, delimiter='\t')
        for row in rd1:
            if row[0] == 'id': continue
            m_data[int(row[0])] = row[1]
    ntcir_cursor_org.execute('SELECT DISTINCT(dataset_id) from ACORDAR_metadata_link_result_TAGME;')
    do_ids = [did for (did, ) in ntcir_cursor_org.fetchall()]
    ntcir_cursor_org.execute('SELECT DISTINCT(dataset_id) from ACORDAR_metadata_info;')
    done_ids = set([did for (did, ) in ntcir_cursor_org.fetchall()])
    for did in do_ids:
        if did in done_ids: continue
        ntcir_cursor_org.execute('insert into ACORDAR_metadata_info (dataset_id, metadata) values (%s, %s)', (did, m_data[did]))
    ntcir_db.commit()

def store_cos_max_score():
    # ntcir_cursor_org.execute('SELECT DISTINCT(eid) from ACORDAR_metadata_entity_freq;')
    
    # done_eids = set()
    # for(eid,) in ntcir_cursor_org.fetchall():
    #     done_eids.add(eid)

    scores = json_load('/home/yourname/code/erm/src/scores.json')
    ind2eid = json_load('/home/yourname/code/erm/src/q_vector_ind2eid.json')

    indexed_dids = json_load('/home/yourname/code/erm/src/indexed_dids.json')

    assert len(scores) == len(indexed_dids), f'{len(scores)} != {len(indexed_dids)}'

    for i, one_eid_scores in enumerate(scores):
        did = indexed_dids[i]
        assert len(one_eid_scores) == len(list(ind2eid.keys())), f'{len(one_eid_scores)} != { len(list(ind2eid.keys()))}'

        for j, score in enumerate(one_eid_scores):
            eid = ind2eid[str(j)]
            sql = 'INSERT IGNORE into ACORDAR_metadata_entity_freq (eid, dataset_id, cos_score) VALUES (%s, %s, %s)'
            ntcir_cursor_org.execute(sql, (eid, did, score))

        ntcir_db.commit()
        print(i, 'done')


if __name__ == '__main__':
    # from configs import link_method, data_field, test_collection_name
    # data_field = 'illusnip'
    # store_ACORDAR_link_res()
    # store_ntcir_file_format()
    # store_link_res(link_method)
    # store_ACORDAR_metadata_info()
    store_cos_max_score()