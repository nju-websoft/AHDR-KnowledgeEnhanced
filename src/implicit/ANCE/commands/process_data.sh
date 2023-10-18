
# ANCE FirstP

seq_length=512
base_data_dir=../data/emnlp2023_data/ntcir2ntcir/
tokenizer_type="roberta-base"

python ../data/msmarco_data.py \
  --data_dir ${base_data_dir} \
  --out_data_dir ${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/ \
  --model_type rdot_nll \
  --model_name_or_path /home2/yourname/Models/roberta-base/ \
  --max_seq_length ${seq_length} \
  --data_type 1