exit 0;

export MODEL=BioLinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL

############################### MedQA ###############################
task=medqa_usmle_hf
datadir=../data/mc/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u mc/run_multiple_choice.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 --warmup_steps 100 --num_train_epochs 2 --max_seq_length 512 --fp16 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
