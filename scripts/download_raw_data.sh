#!/bin/bash
mkdir -p raw_data
echo "Please follow the procedures in this script step by step by hand\n"
exit 0;


############################## Download MRQA ##############################
OUTPUT=raw_data/mrqa/train
mkdir -p $OUTPUT

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz  -O $OUTPUT/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O $OUTPUT/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O $OUTPUT/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O $OUTPUT/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O $OUTPUT/NaturalQuestions.jsonl.gz

gzip -d $OUTPUT/SQuAD.jsonl.gz
gzip -d $OUTPUT/NewsQA.jsonl.gz
gzip -d $OUTPUT/TriviaQA.jsonl.gz
gzip -d $OUTPUT/SearchQA.jsonl.gz
gzip -d $OUTPUT/HotpotQA.jsonl.gz
gzip -d $OUTPUT/NaturalQuestions.jsonl.gz


OUTPUT=raw_data/mrqa/dev
mkdir -p $OUTPUT
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz  -O $OUTPUT/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O $OUTPUT/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O $OUTPUT/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O $OUTPUT/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O $OUTPUT/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O $OUTPUT/NaturalQuestions.jsonl.gz

gzip -d $OUTPUT/SQuAD.jsonl.gz
gzip -d $OUTPUT/NewsQA.jsonl.gz
gzip -d $OUTPUT/TriviaQA.jsonl.gz
gzip -d $OUTPUT/SearchQA.jsonl.gz
gzip -d $OUTPUT/HotpotQA.jsonl.gz
gzip -d $OUTPUT/NaturalQuestions.jsonl.gz



############################## Download MMLU ##############################
mkdir -p raw_data/mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O raw_data/mmlu/data.tar
tar -xf raw_data/mmlu/data.tar -C raw_data/mmlu



################### Download MedQA-USMLE (need to manually follow) ###################
mkdir -p raw_data/medqa
# Follow the following steps.
# 1. As instructed by the author's GitHub page (https://github.com/jind11/MedQA), download the dataset (`data_clean.zip`) from https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view
# 2. Move it to `raw_data/medqa/data_clean.zip` and unzip it.


################### Download BLURB (need to manually follow) ###################
mkdir -p raw_data/blurb
cd raw_data/blurb

wget https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz
tar -xf data_generation.tar.gz

# Then follow the instruction in `data_generation/README.md`.
# At end you will get all the official BLURB datasets under `data_generation/data`.
