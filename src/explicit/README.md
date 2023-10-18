In order to apply explicit knowledge-enhanced retrieval methods, the knowledge base, entity linking tool, entity embedding tool and the evaluation set of the experiment need to be specified in the configs, for example, the following configuration specifies entity linking via TAGME on Wikipedia and entity vector acquisition via Wikipedia2vec:
```python
common_dict_config = {
    'link_method': 'TAGME', # 'TAGME|FALCON2|REL'
    'embedding_method': 'wiki2vec',  # 'wiki2vec|wikidata_rdf2vec_sg_200_1|kgtk',
    'test_collection_name': 'ACORDAR', # 'ACORDAR|NTCIR-E|ntcir16'
    'train_or_test': 'test',
    'data_field': 'metadata', 
    'res_dir_name': 'multi_run',
    'k': 10, # rerank top k
}
```

## Entity Linking
Run `python entity_link.py` for entity linking

## Retrieval Based on Entity Set Similarity
Run `python retrieve.py` for entity similarity estimation