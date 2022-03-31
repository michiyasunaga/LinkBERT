import os
import csv
import json
import random
import shutil
import numpy as np
from tqdm import tqdm


root = "data/mc"
os.system(f"mkdir -p {root}")


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)

def process_mmlu(dname):
    for split in ["dev", "val", "test"]:
        lines = csv.reader(open(f"raw_data/mmlu/data/{split}/{dname}_{split}.csv"))
        outs, lens = [], []
        for i, line in enumerate(lines):
            items = line
            sent1 = items[0]
            outs.append({"id": f"{dname}_{split}-{i:05d}",
                          "sent1": sent1,
                          "sent2": "",
                          "ending0": items[1],
                          "ending1": items[2],
                          "ending2": items[3],
                          "ending3": items[4],
                          "label": ord(items[5]) - ord("A")
                        })
            lens.append(len(sent1) + max([len(items[1]),len(items[2]), len(items[3]), len(items[4])]))
        print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
        os.system(f"mkdir -p {root}/mmlu_hf/{dname}")
        dump_jsonl(outs, f"{root}/mmlu_hf/{dname}/{split}.json")


process_mmlu("professional_medicine")
