# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_en) carefully before running the script

# For training -> default setting, ignore resume_from_checkpoint
# For resuming pretraining -> disable --overwrite_output_dir, add resume_from_checkpoint
# For eval -> disable --deepspeed,  --overwrite_output_dir, add resume_from_checkpoint, change --do_eval -> --do_train

## PEFT part
lr=2e-5

## USER-DEFINED part
## PATHS
pretrained_model=vilm/vinallama-7b
vietnamese_tokenizer_path=vilm/vinallama-7b
dataset_dir=/home/lhson/lhson/text-dedup/output/minhash/
data_cache=/home/lhson/lhson/data/pretrain_data/cache
output_dir=/home/lhson/lhson/Chinese-LLaMA-Alpaca-2/scripts/training/output_dir/test_pretrain
logging_dir=runs/pretrain

# resume_checkpoint_path=/work/ctv.sangdv/vietnamese-llama-ckp/231030/checkpoint-20200

## Training specific
per_device_train_batch_size=16
per_device_eval_batch_size=2
gradient_accumulation_steps=18
block_size=512

export OMP_NUM_THREADS=64

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/lhson/lhson/llm-legal/alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml  run_clm.py  \
    --report_to tensorboard \
    --logging_dir ${logging_dir} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name ${vietnamese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --load_from_disk \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed 73 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --cosine_min_lr_ratio 0.1 \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 5 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 64 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --load_in_kbits 16 \
    --bf16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False  \
    --overwrite_output_dir \

    
    # --modules_to_save ${modules_to_save} \
    # --lora_rank ${lora_rank} \
    # --lora_alpha ${lora_alpha} \
    # --lora_dropout ${lora_dropout} \
    # --trainable ${lora_trainable} \
    # --modules_to_save ${modules_to_save} \
# --resume_from_checkpoint ${resume_checkpoint_path} 
# --overwrite_output_dir 
# tensorboard --logdir runs/ --port 9876
