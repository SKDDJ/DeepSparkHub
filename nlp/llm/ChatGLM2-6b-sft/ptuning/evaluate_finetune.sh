CHECKPOINT=adgen-chatglm2-6b-ft-1e-4
STEP=3000
NUM_GPUS=2

for configfile in configuration_chatglm.py modeling_chatglm.py quantization.py tokenization_chatglm.py
do
    if [[ ! -f output/$CHECKPOINT/checkpoint-$STEP/$configfile ]]; then
        cp data/chatglm2-6b/$configfile output/$CHECKPOINT/checkpoint-$STEP/
    fi
done

deepspeed --num_gpus=$NUM_GPUS main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ./output/$CHECKPOINT/checkpoint-$STEP  \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --fp16_full_eval
