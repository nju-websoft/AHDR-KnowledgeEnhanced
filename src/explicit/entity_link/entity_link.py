import multiprocessing
from multiprocessing import Manager, Pool
import json
import requests
from database import sql_one, sql_list, conn, cursor, ntcir_cursor_org
import os

# from multiprocess import Pool

from logger import log
# from preprocess.extract_file_text import get_file_info, time_limit, TimeoutException
from util import json_load, json_dump

# config.link_method, config.data_field, config.test_collection_name

from configs import common_config as config, link_res_dir_path

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

import urllib3
http = urllib3.PoolManager()
def link_falcon(input):
    fail = False
    try:                    
        
        data = {"text": input}
        encoded_data = json.dumps(data).encode('utf-8')
        r = http.request('POST','https://labs.tib.eu/falcon/falcon2/api?mode=long&k=5',
                        body=encoded_data,
                        headers={'Content-Type': 'application/json'})
        result = json.loads(r.data.decode('utf-8'))

    except Exception as e:
        log.info(f'requests exception raised: {e}')
        fail = True
        return fail, {}
    else:
        if r.status != 200: 
            log.info(f'status_code: {r.status}\nlabels: {input}')
            fail = True
        return fail, result


def call_REL(input, port):
    API_URL = f'http://localhost:{port}' # 'http://gem.cs.ru.nl/api/'
    # API_URL = f'https://rel-entity-linker.d4science.org/'
    print(input)
    fail = False
    try:
        res = requests.post(API_URL, json={
            "text": input,
            "spans": []
        }).json()
        print(res)
    except Exception as e:
        log.info(f'expception raised: {e}')
        res = []
        fail = True
    return fail, res


m_str_dict = {}
d_str_dict = {}

def init_m_str_dict():
    global m_str_dict
    # ntcir_cursor_org.execute(f'select dataset_id, metadata from {config.test_collection_name}_metadata_info')
    ntcir_cursor_org.execute(f'SELECT query_id, query_text FROM `{config.test_collection_name}_query` where type = "test"')
    for (id, m_str) in ntcir_cursor_org.fetchall():
        m_str_dict[id] = m_str
    print('len(m_str_dict): ', len(m_str_dict))

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

# file_info_dict = get_file_info()
chunk_limit = 2**16

import pymysql
conn = None
ntcir_cursor = None

fail_rec_path = f'rel_fail.txt'
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

    if config.link_method == 'TAGME':
        link_result = dict()
        fail, link_result = link_tagme(m_str)
        if fail: log.info(f'fail dealing with dataset {dataset_id}')
        else : # store result into sqlite database
            with open(f'{link_res_dir_path}/{dataset_id}.json', 'w+') as fp:
                json.dump(link_result, fp, indent=2)
            print(dataset_id, 'succ')
    elif config.link_method == 'falcon':
        link_result = dict()
        fail, link_result = link_falcon(m_str)
        if fail: log.info(f'fail dealing with dataset {dataset_id}')
        else : # store result into sqlite database
            with open(f'{link_res_dir_path}/{dataset_id}.json', 'w+') as fp:
                json.dump(link_result, fp, indent=2)
            print(dataset_id, 'succ')
    else:
        free_port = get_free_port(ports)
        # log.info(f"PID {os.getpid()} port: {free_port}")
        print(ports)
        while free_port is None:
            print('no free port, sleep 3s')
            os.system('sleep 3')
            free_port = get_free_port(ports)

        # ports.remove(free_port)
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
    link_res_dir_path = f'/home/yourname/code/erm/data/annos/{config.test_collection_name}/{config.test_collection_name}_illusnip' 
    if dataset_id not in d_str_dict:
        log.info('dataset_id {} not in d_str_dict'.format(dataset_id))
        return
    d_str = d_str_dict[dataset_id]
    if config.link_method == 'TAGME':
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
    # file_path = f'{config.file_texts_path}/{dataset_id}_{file_id}_{data_format}.txt'
    file_path = f"{config.file_texts_path}/{file_id}_{data_filename.replace('.', '_')}.txt"
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
        if config.link_method == 'REL':
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
    if dataset_id in done_dids: return
    if config.data_field == 'metadata': link_dataset_metadata(dataset_id, ports)
    elif config.data_field == 'content':
        if config.test_collection_name == 'ntcir':
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
        # did = int(prefix.split('_')[0])
        did = prefix.split('.')[0]
        # fid = int(prefix.split('_')[1])
        done_dids.add(did)
        # done_fids.add(fid)
        

def get_do_ids():
    from util import get_topk_candidate_set
    dids = list(get_topk_candidate_set(10))

    return dids

if __name__ == '__main__':

    if not os.path.isfile(fail_rec_path): os.system(f'touch {fail_rec_path}')

    init_done_ids() 
    init_m_str_dict()

    do_ids = list(m_str_dict.keys())

    print('len:', len(do_ids))

    manager = Manager()
    all_ports = manager.list()
    for port in range(5555, 5559): all_ports.append(port)
    p = Pool(4) 
    p.starmap(link_dataset, [(did, all_ports) for did in do_ids]) 
    p.close()
    p.join()
    
