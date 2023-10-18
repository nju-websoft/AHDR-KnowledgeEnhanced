# SpanQualifier

## Installation
```angular2html
pip install -r requirements.txt
```

### The Usage of MonoBert
To train the MonoT5 model on NTCIR-E dataset, you can use the following command:
```angular2html
python run_monobert_nticr_e.py --gpu 0 --seed 0 --epoch_num 10 --model_name castorini/monobert-large-msmarco
```

To train the MonoT5 model on ACORDAR dataset, you can use the following command:
```angular2html
python run_monobert_acordar.py --gpu 0 --seed 0 --epoch_num 10 --model_name castorini/monobert-large-msmarco
```
