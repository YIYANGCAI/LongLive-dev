# python feature_extraction.py \
#     --meta_file ./dataset_meta/meta.jsonl \
#     --output_dir /aifs4su/caiyiyang/datasets/longlive_toy_data/stage_1/features

# Distributed (multi-GPU)
NPROC=${NPROC:-2}
torchrun \
    --nproc_per_node=$NPROC \
    feature_extraction.py \
    --meta_file ./dataset_meta/meta.jsonl \
    --output_dir /aifs4su/caiyiyang/datasets/longlive_toy_data/stage_1/features
