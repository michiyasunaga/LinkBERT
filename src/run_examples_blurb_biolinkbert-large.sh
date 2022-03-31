exit 0;

export MODEL=BioLinkBERT-large
export MODEL_PATH=michiyasunaga/$MODEL

############################### QA: PubMedQA ###############################
task=pubmedqa_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --fp16 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 23 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### QA: BioASQ ###############################
task=bioasq_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --fp16 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 30  --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### BIOSSES ###############################
task=BIOSSES_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name pearsonr \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --fp16 \
  --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 100  --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### HoC ###############################
task=HoC_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name hoc \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --fp16 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### RE: ChemProt ###############################
task=chemprot_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 4 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### RE: DDI ###############################
task=DDI_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 3 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### RE: GAD ###############################
task=GAD_hf
datadir=../data/seqcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u seqcls/run_seqcls.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --metric_name PRF1 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 5  --max_seq_length 256 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### EBM PICO ###############################
task=ebmnlp_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --return_macro_metrics \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 1 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &



############################### NER: JNLPBA ###############################
task=JNLPBA_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 --seed 3 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 4 --max_seq_length 512 \
  --weight_decay 0.01 --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt

############################### NER: NCBI-disease ###############################
task=NCBI-disease_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 10 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### NER: BC2GM ###############################
task=BC2GM_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 30 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### NER: BC5CDR-disease ###############################
task=BC5CDR-disease_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --fp16 \
  --learning_rate 3e-5 --warmup_ratio 0.5 --num_train_epochs 20 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &

############################### NER: BC5CDR-chem ###############################
task=BC5CDR-chem_hf
datadir=../data/tokcls/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u tokcls/run_ner.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --fp16 \
  --learning_rate 2e-5 --warmup_ratio 0.5 --num_train_epochs 8 --max_seq_length 512 \
  --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
