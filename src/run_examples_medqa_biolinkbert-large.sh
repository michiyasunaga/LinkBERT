exit 0;

export MODEL=BioLinkBERT-large
export MODEL_PATH=michiyasunaga/$MODEL

############################### MedQA ###############################
task=medqa_usmle_hf
datadir=../data/mc/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 32 \
  --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
  --learning_rate 3e-5 --warmup_steps 500 --num_train_epochs 6 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy steps --eval_steps 100 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### MMLU professional medicine ###############################
# Here we load the LinkBERT that was finetuned on the MedQA task above, and evalute it on the MMLU professional medicine task.
# Specify `model_to_load` to be the model saved from above.
model_to_load=runs/medqa_usmle_hf/$MODEL

task=professional_medicine
datadir=../data/mc/mmlu_hf/$task
outdir="runs/mmlu_${task}_eval/${MODEL}"
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $model_to_load \
  --train_file $datadir/dev.json --validation_file $datadir/val.json --test_file $datadir/test.json \
  --do_predict --per_device_eval_batch_size 4 --max_seq_length 512 --fp16 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
