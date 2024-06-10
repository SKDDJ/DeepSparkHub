export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export dataset_name="lambdalabs/pokemon-blip-captions"
export dataset_name="YaYaB/onepiece-blip-captions"

# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 28888 --mixed_precision bf16 train_text_to_image.py \
accelerate launch --main_process_port 10009 --mixed_precision bf16 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"

  