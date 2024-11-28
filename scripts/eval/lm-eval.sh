#! /bin/bash

WORKING_DIR=$PWD

#---------------------------------------------------------
# Tasks (Specify one or all in TASK below)
#---------------------------------------------------------

FOLDER=l_eval
TASK="race_em,openbookqa_em,boolq_em,hellaswag_em"  

# Run command
echo "Running evaluation on model: $MODEL with task: $TASK"

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL \
    --include_path ./custom \
    --tasks $TASK \
    --device cuda:0 \
    --batch_size auto \
    --gen_kwargs max_new_tokens=20,max_length=None,do_sample=False\
    --num_fewshot 0 \
    --log_samples \
    --output_path ${WORKING_DIR}/playground/${FOLDER} \
