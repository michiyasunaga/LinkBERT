# LinkBERT

This repo provides the model, code & data of our paper: [LinkBERT: Pretraining Language Models with Document Links](https://arxiv.org/abs/2203.15827) (ACL 2022).
```bib
@InProceedings{yasunaga2022linkbert,
  author =  {Michihiro Yasunaga and Jure Leskovec and Percy Liang},
  title =   {LinkBERT: Pretraining Language Models with Document Links},
  year =    {2022},  
  booktitle = {Association for Computational Linguistics (ACL)},  
}
```
### Overview
LinkBERT is a new pretrained language model (improvement of BERT) that captures **document links** such as hyperlinks and citation links to include knowledge that spans across multiple documents. Specifically, it was pretrained by feeding linked documents into the same language model context, besides using a single document as in BERT.

LinkBERT can be used a drop-in replacement for BERT. It achieves better performance for general language understanding tasks (e.g. text classification), and is also particularly effective for **knowledge-intensive** tasks (e.g. question answering) and **cross-document** tasks (e.g. reading comprehension, document retrieval).

<p align="center">
  <img src="./figs/overview.png" width="1000" title="Overview of LinkBERT" alt="">
</p>


## 1. Pretrained Models

We release the pretrained LinkBERT (-base and -large sizes) for both the general domain and biomedical domain. These models have the same format as the [HuggingFace BERT models](https://github.com/huggingface/transformers), and you can easily switch them with LinkBERT models.
<!-- The -base & -large sizes follow the same model configuration as [BERT](https://github.com/google-research/bert). -->


| Model | Size | Domain | Pretraining Corpus | 🤗 Transformers Link |
| ------------- | ------------- | --------- | ---- | ---- |
| LinkBERT-base   | 110M parameters | General | Wikipedia with hyperlinks | [michiyasunaga/LinkBERT-base](https://huggingface.co/michiyasunaga/LinkBERT-base) |
| LinkBERT-large  | 340M parameters | General | Wikipedia with hyperlinks | [michiyasunaga/LinkBERT-large](https://huggingface.co/michiyasunaga/LinkBERT-large) |
| BioLinkBERT-base   | 110M parameters | Biomedicine | PubMed with citation links | [michiyasunaga/BioLinkBERT-base](https://huggingface.co/michiyasunaga/BioLinkBERT-base) |
| BioLinkBERT-large  | 340M parameters | Biomedicine | PubMed with citation links | [michiyasunaga/BioLinkBERT-large](https://huggingface.co/michiyasunaga/BioLinkBERT-large) |

To use these models in 🤗 Transformers:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-large')
model = AutoModel.from_pretrained('michiyasunaga/LinkBERT-large')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
```


To fine-tune the models, see Section 2 & 3 below.
When fine-tuned on downstream tasks, LinkBERT achieves the following results.  
**General benchmarks ([MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) and [GLUE](https://gluebenchmark.com/)):**

|                         | HotpotQA | TriviaQA | SearchQA | NaturalQ | NewsQA   | SQuAD    | GLUE      |
| ----------------------  | -------- | -------- | -------- | -------- | ------   | -----    | --------  |
|                         | F1       | F1       | F1       |  F1      | F1       | F1       | Avg score |
| BERT-base               | 76.0     | 70.3     | 74.2     | 76.5     | 65.7     | 88.7     | 79.2      |
| **LinkBERT-base**       | **78.2** | **73.9** | **76.8** | **78.3** | **69.3** | **90.1** | **79.6**  |
| BERT-large              | 78.1     | 73.7     | 78.3     | 79.0     | 70.9     | 91.1     | 80.7      |
| **LinkBERT-large**      | **80.8** | **78.2** | **80.5** | **81.0** | **72.6** | **92.7** | **81.1**  |


**Biomedical benchmarks ([BLURB](https://microsoft.github.io/BLURB/), [MedQA](https://github.com/jind11/MedQA), [MMLU](https://github.com/hendrycks/test), etc):** BioLinkBERT attains new state-of-the-art 😊

|                         | BLURB score | PubMedQA | BioASQ   | MedQA-USMLE |
| ----------------------  | --------    | -------- | -------  | --------    |
| PubmedBERT-base         | 81.10       | 55.8     | 87.5     | 38.1        |
| **BioLinkBERT-base**    | **83.39**   | **70.2** | **91.4** | **40.0** |
| **BioLinkBERT-large**   | **84.30**   | **72.2** | **94.8** | **44.6** |

|                         | MMLU-professional medicine |
| ----------------------  | --------  |
| GPT-3 (175 params)      | 38.7      |
| UnifiedQA (11B params)  | 43.2      |
| **BioLinkBERT-large (340M params)** | **50.7**  |



## 2. Set up environment and data
### Environment
Run the following commands to create a conda environment:
```bash
conda create -n linkbert python=3.8
source activate linkbert
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.9.1 datasets==1.11.0 fairscale==0.4.0 wandb sklearn seqeval
```

### Data
You can download the preprocessed datasets on which we evaluated LinkBERT from [**[here]**](https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip). Simply download this zip file and unzip it.
This includes:
- [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) question answering datasets (HotpotQA, TriviaQA, NaturalQuestions, SearchQA, NewsQA, SQuAD)
- [BLURB](https://microsoft.github.io/BLURB/) biomedical NLP datasets (PubMedQA, BioASQ, HoC, Chemprot, PICO, etc.)
- [MedQA-USMLE](https://github.com/jind11/MedQA) biomedical reasoning dataset.
- [MMLU-professional medicine](https://github.com/hendrycks/test) reasoning dataset.

They are all preprocessed in the [HuggingFace dataset](https://github.com/huggingface/datasets) format.

If you would like to preprocess the raw data from scratch, you can take the following steps:
- First download the raw datasets from the original sources by following instructions in `scripts/download_raw_data.sh`
- Then run the preprocessing scripts `scripts/preprocess_{mrqa,blurb,medqa,mmlu}.py`.


## 3. Fine-tune LinkBERT

Change the working directory to `src/`, and follow the instructions below for each dataset.

### MRQA
To fine-tune for the MRQA datasets (HotpotQA, TriviaQA, NaturalQuestions, SearchQA, NewsQA, SQuAD), run commands listed in `run_examples_mrqa_linkbert-{base,large}.sh`.

### BLURB
To fine-tune for the BLURB biomedial datasets (PubMedQA, BioASQ, HoC, Chemprot, PICO, etc.), run commands listed in `run_examples_blurb_biolinkbert-{base,large}.sh`.

### MedQA & MMLU
To fine-tune for the MedQA-USMLE dataset, run commands listed in `run_examples_medqa_biolinkbert-{base,large}.sh`.

To evaluate the fine-tuned model additionally on MMLU-professional medicine, run the commands listed at the bottom of `run_examples_medqa_biolinkbert-large.sh`.


### Reproducibility
We also provide [Codalab worksheet](https://worksheets.codalab.org/worksheets/0x7a6ab9c8d06a41d191335b270da2902e), on which we record our experiments. You may find it useful for replicating the experiments using the same model, code, data, and environment.
