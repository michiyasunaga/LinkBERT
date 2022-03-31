import os
import json
import random
import shutil
import numpy as np
from tqdm import tqdm


root = "data/qa"
os.system(f"mkdir -p {root}")


mrqa_raw_files = ["SQuAD.jsonl", "NewsQA.jsonl", "TriviaQA.jsonl", "SearchQA.jsonl", "HotpotQA.jsonl", "NaturalQuestions.jsonl"]
mrqa_dataset_names = ["squad", "newsqa", "triviaqa", "searchqa", "hotpot", "naturalqa"]


def organize_mrqa():
    for dname in mrqa_dataset_names:
        os.system(f"mkdir -p {root}/{dname}")
    for data_file, output_dir in tqdm(zip(mrqa_raw_files, mrqa_dataset_names)):
        os.system(f"cp -rp raw_data/mrqa/train/{data_file} {root}/{output_dir}/train.jsonl")
        os.system(f"cp -rp raw_data/mrqa/dev/{data_file} {root}/{output_dir}/dev_mrqa.jsonl")

organize_mrqa()


def split_dev_mrqa(dname, fname):
    lines = open(f"{root}/{dname}/{fname}.jsonl").readlines()
    lines = lines[1:]
    print ("len(lines)", len(lines))
    split_info = json.load(open(f"scripts/inhouse_splits/inhouse_split_{dname}.json"))
    assert len(split_info["dev"]) + len(split_info["test"]) == len(lines)
    with open(f"{root}/{dname}/dev.jsonl", "w") as outf:
        print (json.dumps({"header": {"dataset": dname, "split": "dev"}}), file=outf)
        for id in split_info["dev"]:
            print (lines[id].strip(), file=outf)
    with open(f"{root}/{dname}/test.jsonl", "w") as outf:
        print (json.dumps({"header": {"dataset": dname, "split": "test"}}), file=outf)
        for id in split_info["test"]:
            print (lines[id].strip(), file=outf)

for dname in mrqa_dataset_names:
    print (dname)
    split_dev_mrqa(dname, "dev_mrqa")


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)

def process_mrqa(dname, fname):
    lines = open(f"{root}/{dname}/{fname}.jsonl").readlines()
    lines = lines[1:]
    outs, lens = [], []
    for line in lines:
        paragraph = json.loads(line)
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            try:
                id = qa["id"]
            except:
                id = qa["qid"]
            question = qa["question"].strip()
            answers = []
            answer_starts = []
            for elm in qa["detected_answers"]:
                answer = elm["text"]
                answer_start = elm["char_spans"][0][0]
                answer_end = elm["char_spans"][0][1]
                answers.append(answer)
                answer_starts.append(answer_start)
            outs.append({"id": id, "question": question, "context": context, "answers": {"answer_start": answer_starts, "text": answers}})
            lens.append(len(question) + len(context))
    print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)))
    #
    os.system(f"mkdir -p {root}/{dname}_hf")
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")


for dname in mrqa_dataset_names:
    for fname in ["train", "dev", "test"]:
        print (dname, fname)
        process_mrqa(dname, fname)
