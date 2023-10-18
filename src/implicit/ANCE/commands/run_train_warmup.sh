# This script is for training the warmup checkpoint for ANCE
export  CUDA_VISIBLE_DEVICES=3

data_dir="../data/emnlp2023_data/ntcir16/"

output_dir="../output/from_ms_marco/ntcir16/"
cmd="python3 -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 ../drivers/run_warmup.py --train_model_type rdot_nll \
  --model_name_or_path /home2/yourname/Models/ance-msmarco-passage \
  --task_name MSMarco --do_train --evaluate_during_training --data_dir ${data_dir}  --max_seq_length 128     --per_gpu_eval_batch_size=256 \
  --per_gpu_train_batch_size=32       --learning_rate 2e-4  --logging_steps 1000   --num_train_epochs 2.0   --output_dir ${output_dir} \
  --warmup_steps 1000  --overwrite_output_dir --save_steps 30000 --gradient_accumulation_steps 1  --expected_train_size 35000000 --logging_steps_per_eval 20 \
  --fp16 --optimizer lamb --log_dir ~/tensorboard/${DLWS_JOB_ID}/logs/OSpass "

echo $cmd
eval $cmd
