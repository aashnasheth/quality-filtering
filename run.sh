#!/bin/bash
conda init
conda activate ideology
export HF_DATASETS_CACHE="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/" #~/../../../gscratch/ark/aashnas
export TRANSFORMERS_CACHE="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/"
export HF_HOME="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/"

# STEP 0 train classifiers
# python Create_Classifier.py
# python Create_Toxicity_Prompts.py
# python Doc_Similarity_copy.py
python Compare_Classifiers.py --ss_thres 0.5 --thres 0.5
# python Compare_Classifiers.py --ss_thres 0.7 --thres 0.5
# python Compare_Classifiers.py --ss_thres 0.7 --thres 0.3

# python Compare_Classifiers.py --ss_thres 0.5 --thres 0.9
# python Compare_Classifiers.py --ss_thres 0.5 --thres 0.7
# python Compare_Classifiers.py --ss_thres 0.5 --thres 0.3

# python Compare_Classifiers.py --ss_thres 0.3 --thres 0.9
# python Compare_Classifiers.py --ss_thres 0.3 --thres 0.7
# python Compare_Classifiers.py --ss_thres 0.3 --thres 0.5


# STEP 1 select train tokens and validation tokens
# python Select_100K_tokens_val.py > reddit_val_dataset_creation_log.txt
# python Select_1B_tokens_parallel.py
# python Select_100K_tokens_val.py > reddit_val_dataset_creation_log.txt
# python Select_100K_tokens.py

# STEP 3 train the model
# # from here: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
# # --max_steps 300000      --eval_steps 250
# export HF_DATASETS_CACHE="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/" #~/../../../gscratch/ark/aashnas
# export TRANSFORMERS_CACHE="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/"
# export HF_HOME="/mmfs1/home/aashnas/../../../gscratch/ark/aashnas/"
# increase batch size (> batch size, find largest to fit in gpu)
# sequence length is block size
# python run_clm.py     --model_name_or_path gpt2      --train_file c4_baseline_train_1b.txt     --do_train     --validation_file c4_baseline_val_100k.txt     --do_eval     --output_dir ./tmp/baseline_model     --per_device_eval_batch_size 32 --per_device_train_batch_size 32    --block_size 512    --max_steps 16000      --eval_steps 150         --evaluation_strategy steps --save_steps 32
# python run_clm.py     --model_name_or_path gpt2      --train_file c4_ss_train_1b.txt     --do_train     --validation_file c4_ss_val_100k.txt     --do_eval     --output_dir ./tmp/ss_model     --per_device_eval_batch_size 32 --per_device_train_batch_size 32    --block_size 512    --max_steps 160000      --eval_steps 150        --evaluation_strategy steps --save_steps 32

# STEP 3.5 sanity check by do-eval on 3 eval datasets  
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file dolma_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file dolma_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file dolma_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file c4_ss_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file c4_ss_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file c4_ss_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "WikiText"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file c4_baseline_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file c4_baseline_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file c4_baseline_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "PalomaAA"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_aa_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_aa_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_aa_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "PalomaWhite"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_white_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_white_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_white_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "PalomaGab"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_gab_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_gab_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_gab_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "PalomaFalcon"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_falcon_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_falcon_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_falcon_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "Subreddits"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_subreddits_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_subreddits_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_subreddits_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "pes2o"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file pes2o_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file pes2o_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file pes2o_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "reddit"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file reddit_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file reddit_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file reddit_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512

# echo "wiki"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file wiki_val_200k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file wiki_val_200k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file wiki_val_200k.txt     --do_eval     --output_dir ./tmp/evals --per_device_eval_batch_size 32     --block_size 512

# echo "books"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file books_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file books_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file books_val_100k.txt     --do_eval     --output_dir ./tmp/evals --per_device_eval_batch_size 32     --block_size 512

# echo "4chan"
# python run_clm.py     --model_name_or_path ./tmp/ss_model/checkpoint-3648    --validation_file paloma_4chan_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path ./tmp/baseline_model/checkpoint-3648    --validation_file paloma_4chan_val_100k.txt     --do_eval     --output_dir ./tmp/evals     --per_device_eval_batch_size 32     --block_size 512
# python run_clm.py     --model_name_or_path gpt2    --validation_file paloma_4chan_val_100k.txt     --do_eval     --output_dir ./tmp/evals --per_device_eval_batch_size 32     --block_size 512

# python Evaluate_toxicity.py     --model ./tmp/ss_model/checkpoint-3648
# python Evaluate_toxicity.py     --model ./tmp/baseline_model/checkpoint-3648
# python Evaluate_toxicity.py     --model gpt2
# python Evaluate_toxicity.py

# python -m scripts.run_toxicity_experiment \
#     --use-dataset \
#     --dataset-file ../quality-filter/toxic_prompts.jsonl \
#     --model-type gpt2 \
#     --model ../quality-filter/tmp/baseline_model/checkpoint-3648 \
#     --perspective-rate-limit 1000 \
#     --alpha 2.0 \
#     --filter_p 0.9 \
#     outputs

# python Compare_Classifiers.py

# STEP 4 test toxicity of model
# python Evaluate_toxicity_copy.py --model gpt2 --output perspective_gpt2_25_1.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/baseline_model/checkpoint-3648 --output perspective_baseline_25_1.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/ss_model/checkpoint-3648 --output perspective_ss_25_1.jsonl

# python Evaluate_toxicity_copy.py --model gpt2 --output perspective_gpt2_25_2.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/baseline_model/checkpoint-3648 --output perspective_baseline_25_2.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/ss_model/checkpoint-3648 --output perspective_ss_25_2.jsonl

# python Evaluate_toxicity_copy.py --model gpt2 --output perspective_gpt2_25_3.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/baseline_model/checkpoint-3648 --output perspective_baseline_25_3.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/ss_model/checkpoint-3648 --output perspective_ss_25_3.jsonl

# python Evaluate_toxicity_copy.py --model gpt2 --output perspective_gpt2_25_4.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/baseline_model/checkpoint-3648 --output perspective_baseline_25_4.jsonl
# python Evaluate_toxicity_copy.py --model ../quality-filter/tmp/ss_model/checkpoint-3648 --output perspective_ss_25_4.jsonl
# python Evaluate_toxicity_test.py --model gpt2

python Doc_Similarity_toxicity.py