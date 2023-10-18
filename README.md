# An Empirical Investigation of Implicit and Explicit Knowledge-Enhanced Methods for Ad Hoc Dataset Retrieval

This repository holds the scripts, retrieval source code and experimental outputs corresponding to the tables in the paper "An Empirical Investigation of Implicit and Explicit Knowledge-Enhanced Methods for Ad Hoc Dataset Retrieval", to facilitate the reproduction of the experimental results of the paper.

## Requirements
This repo uses Python 3.7+ and Pytorch 1.9+ and the [Hugging Face Transformers](https://github.com/huggingface/transformers) library. The following lists a subset of required packages.

- torch
- transformers
- faiss-cpu/faiss-gpu
- pandas
- tqdm

## Test Collections
We conducted experiments on the following two test collections for ad hoc dataset retrieval:
### NTCIR-E
[NTCIR-E](https://ntcir.datasearch.jp/data_search_1/) is the English version of the test collection used in the NTCIR-15 Dataset Search task, including 46,615 datasets and 192 queries.
The metadata for all datasets can be downloaded [here](https://drive.google.com/file/d/1mW_FvRGZiBHz4ai42NtVcH5sXTqWBvMe/view?usp=drive_link).

### ACORDAR
[ACORDAR](https://github.com/nju-websoft/ACORDAR) is a test collection specifically over RDF datasets, including 31,589 datasets and 493 queries.
The metadata for all datasets can be downloaded [here](https://github.com/nju-websoft/ACORDAR/raw/main/Data/datasets.json).

## Implicit Knowledge-Enhanced Retrieval
We have placed the retrieval code for all implicit knowledge-enhanced retrieval methods ([monoBERT](https://huggingface.co/castorini/monobert-large-msmarco), [monoT5](https://huggingface.co/castorini/monot5-large-msmarco), [co-Condenser](https://huggingface.co/Luyu/co-condenser-marco), [ColBERT](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz), [ANCE](https://huggingface.co/castorini/ance-msmarco-passage)) in `src/implicit`, each PLM corresponds to a folder and the way to run them can be found in the respective README.

## Explicit Knowledge-Enhanced Retrieval
For the explicit knowledge-enhanced retrieval methods, we used two knowledge bases, Wikipedia and Wikidata. for Wikipedia, we compared two entity linking tools, [TAGME](https://tagme.d4science.org/tagme/) and [REL](https://github.com/informagi/REL), and used [Wikiepdia2vec](https://wikipedia2vec.github.io/wikipedia2vec/) to obtain vectors of entities to calculate cosine similarity; for Wikidata, we used [Falcon2.0](https://github.com/SDM-TIB/falcon2.0) as the entity linking tool and compared the experimental results of entity vectors obtained by [KGTK](https://github.com/usc-isi-i2/kgtk-similarity) and [RDF2vec](https://data.dws.informatik.uni-mannheim.de/rdf2vec/models/Wikidata/4depth/skipgram/).

The code of the explicit knowledge-enhanced retrieval methods is placed in `src/explicit`, you can refer to the instructions in the [README]() to reproduce the experiment.

## Interpolation of Different Retrieval Methods
We use score interpolation (i.e., taking the arithmetic mean of the scores of the different retrieval methods after min-max normalization) to incorporate different retrieval methods, and the script is placed in `src/interpolation`, which can be reproduced by referring to the instructions in [README]().

## Experimental Results and Evaluation
We put all the retrieval results files corresponding to the tables in the paper in `xxx`, which can be reproduced under the guidelines of the above instructions. The result files are json files in the following format:
```json
{
  "query_id1": [
    [
      dataset_id1,
      score1
    ],
    [
      dataset_id2,
      score2
    ],
    ...
  ],
  ...
}
```

To evaluate the results, we use two evaluation metrics: normalized discounted cumulative gain (NDCG) and mean average precision (MAP). We provide a evaluation script for obtaining NDCG@5, NDCG@10, MAP@5 and MAP@10, which takes as input a file in the above json format and outputs a json file in the following format: 
```json
{
  "each query": {
    "1": {
      "ndcg_cut_5": xxx,
      "ndcg_cut_10": xxx,
      "map_cut_5": xxx,
      "map_cut_10": xxx
    },
    ...
  },
  "mean_all": {
    "ndcg_cut_5_mean": xxx,
    "ndcg_cut_10_mean": xxx,
    "map_cut_5_mean": xxx,
    "map_cut_10_mean": xxx
  }
}
```
It is worth noting that since the BM25 method retrieves no results for ten queries on ACORDAR, resulting in only 483 queries in the BM25 results file (and the results files for the reranking of it), we multiply the retrieval results by 483/493 in the script to align with the metrics in the original paper.


## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation