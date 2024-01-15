# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
## PEFT part
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=bkai-foundation-models/vietnamese-llama2-7b-120GB
vietnamese_tokenizer_path=bkai-foundation-models/vietnamese-llama2-7b-120GB
dataset_dir=/home/lhson/lhson/Chinese-LLaMA-Alpaca-2/data_llama/mix
## Training specific
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=18 
max_seq_length=1024
logging_dir=runs/231222_only_alpaca
output_dir=output_dir/vinataba
validation_file=/home/lhson/lhson/Chinese-LLaMA-Alpaca-2/data_llama/eval/alpaca_translated_son_eval_100.json

## DEEPSPEED?
export OMP_NUM_THREADS=64
deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nnodes 1 --nproc_per_node 3 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --full_finetuning False \
    --use_auth_token \
    --report_to tensorboard \
    --logging_dir ${logging_dir} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name ${vietnamese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 73 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 50 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 64 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --bf16 \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
