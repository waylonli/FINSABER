#!/bin/bash

export OPENAI_API_KEY="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"
export PYTHONPATH=$(pwd)
# gpt
# train
 python run.py sim \
 -mdp data/03_model_input/synthetic_dataset.pkl \
 -st 2022-04-14 \
 -et 2022-06-19 \
 -rm train \
 -cp config/tsla_gpt_config.toml \
 -rp data/05_train_model_output
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \

# # train-checkpoint
# python run.py sim-checkpoint \
# -ckp /workspace/FinMem-LLM-StockTrading/data/06_train_checkpoint \
# -rp data/05_train_model_output \
# -cp config/tsla_gpt_config.toml \
# -rm train


# # test
python run.py sim \
-mdp data/finmem_data/03_model_input/synthetic_dataset.pkl \
-st 2022-06-20 \
-et 2022-08-01 \
-rm test \
-cp config/tsla_gpt_config.toml \
-tap  ./data/06_train_checkpoint  \
-ckp ./data/08_test_checkpoint \
-rp ./data/09_results
# # test-checkpoint
# python run.py sim-checkpoint \
# -rm test \
# -ckp ./data/08_test_checkpoint \
# -rp ./data/09_results

python save_file.py
