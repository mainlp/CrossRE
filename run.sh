#!/bin/bash

DATA_PATH=crossre_data
EXP_PATH=<PATH TO EXPERIMENT FOLDER>

DOMAIN="ai"
# domains: news politics science music literature ai
LM='bert-base-cased'
SEEDS=( 4012 5096 8878 8857 9908 )


#iterate over seeds
for rs in "${!SEEDS[@]}"; do
  echo "Experiment on random seed ${SEEDS[$rs]}."

  exp_dir=$EXP_PATH/rs${SEEDS[$rs]}
  # check if experiment already exists
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  # if experiment is new, train classifier
  else
    echo "Training model on random seed ${SEEDS[$rs]}."

    # train
    python3 main.py \
            --train_path $DATA_PATH/${DOMAIN}-train.json \
            --dev_path $DATA_PATH/${DOMAIN}-dev.json \
            --exp_path ${exp_dir} \
            --language_model ${LM} \
            --seed ${SEEDS[$rs]}

    # save experiment info
    echo "Model with RS: ${SEEDS[$rs]}" > $exp_dir/experiment-info.txt
  fi

  # check if prediction already exists
  if [ -f "$exp_dir/${DOMAIN}-test-pred.csv" ]; then
    echo "[Warning] Prediction '$exp_dir/${DOMAIN}-test-pred.csv' already exists. Not re-predicting."

  # if no prediction is available, run inference
  else
    # prediction
    python3 main.py \
            --train_path $DATA_PATH/${DOMAIN}-train.json \
            --test_path $DATA_PATH/${DOMAIN}-test.json \
            --exp_path ${exp_dir} \
            --language_model ${LM} \
            --seed ${SEEDS[$rs]} \
            --prediction_only
  fi

  # check if summary metric scores file already exists
  if [ -f "$EXP_PATH/summary-exps.txt" ]; then
    echo "RS: ${SEEDS[$rs]}" >> $EXP_PATH/summary-exps.txt
  else
    echo "Domain ${DOMAIN}" > $EXP_PATH/summary-exps.txt
    echo "RS: ${SEEDS[$rs]}" >> $EXP_PATH/summary-exps.txt
  fi

  # run evaluation
  python3 evaluate.py \
          --gold_path ${DATA_PATH}/${DOMAIN}-test.json \
          --pred_path ${exp_dir}/${DOMAIN}-test-pred.csv \
          --out_path ${exp_dir} \
          --summary_exps $EXP_PATH/summary-exps.txt

done