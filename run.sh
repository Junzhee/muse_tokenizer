#! /bin/bash


export CUDA_VISIBLE_DEVICES=0


python tokenize_images.py \
    --dataset_path='path of the laion 2b' \
    --output_file='./laion2b_tokenizer.jsonl' \
    --batch_size=256