import multiprocessing
from multiprocessing import Manager, Pool
import json
import requests
import os

# from multiprocess import Pool

from logger import log
from database import sql_one, sql_list, conn, cursor, ntcir_cursor_org
# from preprocess.extract_file_text import get_file_info, time_limit, TimeoutException
from util import json_load, json_dump

# from configs import common_config, format_mapping, file_texts_path, link_res_dir_path

def link_tagme(input):
    token = '02bdcbc7-54bc-4b86-8029-979a40e0c184-843339462' # e25a2d7a-6d5f-47d6-b661-f682ca6d566b-843339462
    url = f'https://tagme.d4science.org/tagme/tag?lang=en&gcube-token={token}' 
    headers = {
        'accept': 'application/json',
        'User-Agent': 'My User Agent 1.0'
    }
    fail = False
    try:                    
        link_result = requests.post(url, data={'text': input}, headers=headers)
    except Exception as e:
        log.info(f'requests exception raised: {e}')
        fail = True
        return fail, {}
    else:
        if link_result.status_code != 200: 
            log.info(f'status_code: {link_result.status_code}\nlabels: {input}')
            fail = True
        return fail, link_result.json()


def call_REL(input, port):
    API_URL = f'http://localhost:{port}' # 'http://gem.cs.ru.nl/api/'

    fail = False
    try:
        res = requests.post(API_URL, json={
            "text": input,
            "spans": []
        }).json()
    except Exception as e:
        log.info(f'expception raised: {e}')
        res = []
        fail = True
    return fail, res


m_str_dict = {}
d_str_dict = {}

def init_m_str_dict():
    global m_str_dict
    if common_config.test_collection_name == 'ACORDAR':
        ntcir_cursor_org.execute(f'select dataset_id, metadata from {common_config.test_collection_name}_metadata_info')
        for (id, m_str) in ntcir_cursor_org.fetchall():
            m_str_dict[id] = m_str_dict
    else:
        for (id, m_str) in sql_list('select dataset_id, metadata from dataset_metadata'):
            m_str_dict[id] = m_str

def get_ACORDAR_metadata_illusnip():
    global m_str_dict
    global d_str_dict
    import csv
    metadata_path = '/home/yourname/code/Scripts-for-DPR/file/ctxs_for_embedding/metadata.tsv'
    illusnip_path = '/home/yourname/code/Scripts-for-DPR/file/ctxs_for_embedding/content_illusnip.tsv'
    m_data = {}
    with open(metadata_path, 'r') as fp1:
        rd1 = csv.reader(fp1, delimiter='\t')
        for row in rd1:
            if row[0] == 'id': continue
            m_data[int(row[0])] = row[1]
    m_str_dict = m_data
    d_data = {}
    with open(illusnip_path, 'r') as fp1:
        rd1 = csv.reader(fp1, delimiter='\t')
        for row in rd1:
            if row[0] == 'id': continue
            d_data[int(row[0])] = row[1]
    d_str_dict = d_data

def link_query():
    q_train_topic_path = '/home/yourname/code/erm/file/data_search_e_train_topics.tsv'
    q_test_topic_path = '/home/yourname/code/erm/file/ntcir_file/data_search_e_test_topics.tsv'
    q_link_res = {}
    with open(q_test_topic_path, 'r') as fp:
        for line in fp.readlines():
            hash_id, q_text = line.split('\t')
            fail, link_result = link_tagme(q_text)
            if fail: log.info(f'fail dealing with query {q_text}')
            else:
                q_link_res[hash_id] = link_result['annotations']
            log.info(f'{q_text} done')
    with open('../file/test_topic_link_result.json', 'w+') as fpw:
        json.dump(q_link_res, fpw)

file_info_dict = get_file_info()
chunk_limit = 2**16

import pymysql
conn = None
ntcir_cursor = None


port_lock = multiprocessing.Lock()
file_lock = multiprocessing.Lock() 
read_rdf_chunk = multiprocessing.Lock()



def get_free_port(free_ports):
    port = None
    with port_lock:
        if len(free_ports) > 0: port = free_ports.pop()
    return port

def add_port(port, free_ports):
    with port_lock:
        free_ports.append(port)

def link_dataset_metadata(dataset_id, ports):
    m_str = m_str_dict[dataset_id] # sql_one(f'select metadata from dataset_metadata where dataset_id = {dataset_id}')[0]
    
    if common_config.link_method == 'TAGME':
        link_result = dict()
        fail, link_result = link_tagme(m_str)
        if fail: log.info(f'fail dealing with dataset {dataset_id}')
        else : # store result into sqlite database
            with open(f'{link_res_dir_path}/{dataset_id}.json', 'w+') as fp:
                json.dump(link_result, fp, indent=2)
            print(dataset_id, 'succ')
    else:
        free_port = get_free_port(ports)
        # log.info(f"PID {os.getpid()} port: {free_port}")
        while free_port is None:
            os.system('sleep 3')
            free_port = get_free_port(ports)
        file_fail, link_result = call_REL(m_str, free_port) # link_tagme(text) # call_REL(text)
        add_port(free_port, ports)

        
        if file_fail:
            log.info(f'{dataset_id}\tlink fail')
            err_msg = 'link fail'
        else:
            # res_dir_path = '/home/yourname/code/erm/data/annos/metadata_REL'
            with open(f'{link_res_dir_path}/{dataset_id}.json', 'w+') as fp:
                json.dump(link_result, fp, indent=2)

def link_dataset_illusnip(dataset_id):
    link_res_dir_path = f'/home/yourname/code/erm/data/annos/{common_config.test_collection_name}/{common_config.test_collection_name}_illusnip' 
    if dataset_id not in d_str_dict:
        log.info('dataset_id {} not in d_str_dict'.format(dataset_id))
        return
    d_str = d_str_dict[dataset_id]
    if common_config.link_method == 'TAGME':
        link_result = dict()
        fail, link_result = link_tagme(d_str)
        if fail: log.info(f'fail dealing with dataset {dataset_id}')
        else: # store result into sqlite database
            with open(f'{link_res_dir_path}/{dataset_id}.json', 'w+') as fp:
                json.dump(link_result, fp, indent=2)
            print(dataset_id, 'succ')


def link_rdf(dataset_id):
    rdf_fail = False
    file_path = f'/home/yourname/code/erm/data/acordar_content/{dataset_id}.json'
    link_res = None
    err_msg = str()
    chunk_id = 0 # now only link first chunk
    if not os.path.isfile(file_path):
        log.info(f'{file_path} is not valid, return')
        return
    try:
        chunk_data = json_load(file_path)
        if len(chunk_data) == 0:
            log.info(f'{file_path} is empty, return')
            return
        text = chunk_data[0]['rdf_terms']
        TEXT_LEN_LIMIT = int(1e4) # 1e6
        if len(text) > TEXT_LEN_LIMIT:
            text = text[:TEXT_LEN_LIMIT]
            log.info('%s too large, make cut', file_path)
        if text.isspace():
            log.info('%s is space, return', file_path)
            return
    except Exception as e:
        file_fail = True
        log.info(f'{dataset_id}\t{chunk_id}\texception: {e}')
        err_msg = e
    else: 
        file_fail, link_result = link_tagme(text)
        if file_fail:
            log.info(f'{dataset_id}\t{chunk_id}\tlink fail')
            err_msg = 'link fail'
        else: link_res = link_result                                                                        
    if file_fail:
        with file_lock:
            with open(fail_rec_path, 'a') as fp:
                fp.write(f'{dataset_id}\t{chunk_id}\t{err_msg}\n')
    else:
        with open(f'{link_res_dir_path}/{dataset_id}_{chunk_id}.json', 'w+') as fpw:
            json.dump(link_res, fpw, indent=2)
        log.info(f'succ: {dataset_id}\t{chunk_id}')



def link_file(dataset_id, file_id, info_dict, ports):
    file_fail = False
    data_format, data_filename = info_dict['data_format'], info_dict['data_filename']
    data_format = format_mapping(data_format)
    # if data_format == 'pdf': return
    log.info(f'linking {dataset_id} {file_id} {data_format}')   
    # file_path = f'{common_config.file_texts_path}/{dataset_id}_{file_id}_{data_format}.txt'
    file_path = f"{common_config.file_texts_path}/{file_id}_{data_filename.replace('.', '_')}.txt"
    if not os.path.isfile(file_path):
        log.info('no file: %s, return', file_path)
        return

    if os.path.getsize(file_path) == 0:
        log.info('file %s size is 0, return', file_path)
        return

    link_res = None
    err_msg = str()
    try:
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        text = ' '.join(lines)
        TEXT_LEN_LIMIT = int(1e4) # 1e6
        if len(text) > TEXT_LEN_LIMIT:
            text = text[:TEXT_LEN_LIMIT]
            log.info('%s too large, make cut', file_path)
        if text.isspace():
            log.info('%s is space, return', file_path)
            return
    except Exception as e:
        file_fail = True
        log.info(f'{dataset_id}\t{file_id}\texception: {e}')
        err_msg = e
    else:    
        if common_config.link_method == 'REL':
            free_port = get_free_port(ports)
            # log.info(f"PID {os.getpid()} port: {free_port}")
            while free_port is None:
                os.system('sleep 3')
                free_port = get_free_port(ports)
            file_fail, link_result = call_REL(text, free_port) # link_tagme(text) # call_REL(text)
            add_port(free_port, ports)
        else: file_fail, link_result = link_tagme(text)
        
        if file_fail:
            log.info(f'{dataset_id}\t{file_id}\tlink fail')
            err_msg = 'link fail'
        else: link_res = link_result                                                                        
    if file_fail:
        with file_lock:
            with open(fail_rec_path, 'a') as fp:
                fp.write(f'{dataset_id}\t{file_id}\t{err_msg}\n')
    else:
        with open(f'{link_res_dir_path}/{dataset_id}_{file_id}.json', 'w+') as fpw:
            json.dump(link_res, fpw, indent=2)
        log.info(f'succ: {dataset_id}\t{file_id}')

def link_dataset(dataset_id, ports):
    global done_dids, done_fids, unlink_fids
    if common_config.data_field == 'metadata': link_dataset_metadata(dataset_id, ports)
    elif common_config.data_field == 'content':
        if common_config.test_collection_name == 'ntcir':
            file_dict = file_info_dict[dataset_id]
            for (file_id, info_dict) in file_dict.items():
                if file_id in done_fids: continue
                if file_id not in unlink_fids: continue
                link_file(dataset_id, file_id, info_dict, ports) 
        else:
            link_rdf(dataset_id)     
    else:
        print('wrong para: datatype')
        exit(0)

done_dids = set()
done_fids = set()
from tmp import get_unlink_fids
unlink_fids = get_unlink_fids()
def init_done_ids():
    global done_dids, done_fids
    for fname in os.listdir(link_res_dir_path):
        prefix = fname.split('.')[0]
        did = int(prefix.split('_')[0])
        fid = int(prefix.split('_')[1])
        done_dids.add(did)
        done_fids.add(fid)



        
        

def get_do_ids():
    # ntcir_cursor_org.execute("SELECT DISTINCT(dataset_id) from ntcir_file_info WHERE detect_format not like 'pdf';")
    ntcir_cursor_org.execute('SELECT dataset_id from ntcir_file_info WHERE file_id in (90217, 6633, 71981, 72403, 88560, 7660, 81911, 45712, 32191, 41446, 8290);')
    dids = []
    for (did, ) in ntcir_cursor_org.fetchall():
        dids.append(did)
    return dids

if __name__ == '__main__':
    
    
    
    fail_rec_path = f'/home/yourname/code/erm/src/entity_link/{common_config.test_collection_name}_link_{common_config.data_field}_fail_{common_config.link_method}.txt'
    if not os.path.isfile(fail_rec_path): os.system(f'touch {fail_rec_path}')

    init_done_ids() 

    # get_ACORDAR_metadata_illusnip()
    init_m_str_dict()

    
    from preprocess.get_sparse_result import get_sparse_res_dids
    # do_ids = get_sparse_res_dids()
    do_ids = get_do_ids()

    do_ids = [id for id in do_ids] # if id not in done_dids]
    print('len:', len(do_ids))    

    # from get_undo_ids import get_undo_dids
    # do_ids = list(get_undo_dids())

    all_ports = [5555, 5556]
    for id in do_ids: #[:10]:  #[:10]:
        print(id)
        # link_dataset_illusnip(id)
        link_dataset(id, all_ports)

    # manager = Manager()
    # all_ports = manager.list()
    # for port in range(5555, 5557): all_ports.append(port)
    # p = Pool(16) # processes=4, initializer=init)
    # p.starmap(link_dataset, [(did, all_ports) for did in do_ids]) 
    # p.close()
    # p.join()
    