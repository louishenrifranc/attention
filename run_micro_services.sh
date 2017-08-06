#!/usr/bin/env bash


OUTPUT_DIR="transformer_output"
TASK_CONFIG="attention/services/create_copy_task/default_config.json"
METADATA="attention/services/attention_train/default_metadata.json"
TRAIN_CONFIG="attention/services/attention_train/default_config.json"
TRAIN_FOLDER="/train"
EVAL_FOLDER="/eval"
MODEL_FOLDER="/model"

echo "Running Task Data Generation"
python3 attention/services/create_copy_task/create_copy_task.py --output_dir $OUTPUT_DIR --config $TASK_CONFIG
echo "Run Task Data Generation"

echo "Run Attention Traininig"
python3 attention/services/attention_train/attentiontrain.py --train_data_dir $OUTPUT_DIR$TRAIN_FOLDER --metadata $METADATA --valid_data_dir $OUTPUT_DIR$EVAL_FOLDER --output_dir $OUTPUT_DIR$MODEL_FOLDER -c $TRAIN_CONFIG
