exit 0;

export MODEL=LinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL

############################### HotpotQA ###############################
task=hotpot_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &


############################### NaturalQuestions ###############################
task=naturalqa_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &


############################### TriviaQA ###############################
task=triviaqa_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &


############################### SQuAD ###############################
task=squad_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &


############################### NewsQA ###############################
task=newsqa_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &


############################### SearchQA ###############################
task=searchqa_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/run_qa.py --model_name_or_path $MODEL \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 3e-5 --num_train_epochs 1 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
