# inference code for long video generation
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  inference.py \
  --config_path configs/longlive_inference.yaml