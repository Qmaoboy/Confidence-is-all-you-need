#!/bin/bash


# python clean_natual_qa_dataset_for_rag.py

# Define datasets, strategies, and common parameters
datasets=("din0s/asqa" "natural_questions" "gsm8k")
strategies=("vanilla" "cot")
api_model="gpt-3.5-turbo-0125"
sim_model="gpt-3.5-turbo-0125"
tasks=("Long_QA" 'QA' "QA")
acc_models=("bertscore" "f1")
data_prompt_amount=10
train_batch_size=50
eval_batch_size=100

# Loop over datasets, strategies, and acc_models
for dataset_index in ${!datasets[@]}; do
    dataset=${datasets[$dataset_index]}
    task=${tasks[$dataset_index]}

#   for strategy in "${strategies[@]}"; do
#         for acc_model in "${acc_models[@]}"; do
#         if [ "$dataset" == "gsm8k" ]; then
#             data_prompt_amount=$gsm8k_data_prompt_amount
#             train_batch_size=$gsm8k_train_batch_size
#         else
#             data_prompt_amount=10
#             train_batch_size=50
#         fi

#     command="python pipeline.py --qa_dataset $dataset --api_model $api_model --strategy $strategy --sim_model $sim_model --task $task --acc_model $acc_model --data_prompt_amount $data_prompt_amount --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size"
#     $command
#         done
#     done
done
# ## din0s/asqa

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model bertscore --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy cot --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model bertscore --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy cot --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset din0s/asqa --api_model gpt-3.5-turbo-0125 --Stretagy cot --sim_model gpt-3.5-turbo-0125 --task Long_QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# # ## natural_questions

# python pipeline.py --qa_dataset natural_questions --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task QA --acc_model bertscore --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset natural_questions --api_model gpt-3.5-turbo-0125 --Stretagy cot --sim_model gpt-3.5-turbo-0125 --task QA --acc_model bertscore --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset natural_questions --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100

# python pipeline.py --qa_dataset natural_questions --api_model gpt-3.5-turbo-0125 --Stretagy cot --sim_model gpt-3.5-turbo-0125 --task QA --acc_model f1 --data_prompt_amount 10 --train_batch_size 50 --eval_batch_size 100


## gsm8k

# python pipeline.py --qa_dataset gsm8k --api_model gpt-3.5-turbo-0125 --Stretagy vanilla --sim_model gpt-3.5-turbo-0125 --task QA --acc_model bertscore --data_prompt_amount 99 --train_batch_size 1 --eval_batch_size 50

