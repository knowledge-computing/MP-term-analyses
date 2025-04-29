import os 
from transformers import BertTokenizerFast
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import polars as pl
import json
import numpy as np
from ast import literal_eval
from pynvml import *

import torch
import gc
from datasets import load_dataset
from transformers import (BertTokenizerFast, AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)
import evaluate

import warnings
from polars.exceptions import MapWithoutReturnDtypeWarning
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)

torch.cuda.empty_cache()
# accelerator.free_memory()
gc.collect()

# DATASET_ID = "disham993/ElectricalNER"
MODEL_ID = "answerdotai/ModernBERT-large"
path_dir = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/formatted/splitted/_regular/'

label2id = {'O':0, 'B-race':1, 'I-race':2}
id2label = {0:'O', 1:'B-race', 2:'I-race'}

dict_to_be = {}

def print_gpu_utilization() -> None:
    """
    Print current GPU utilization stat
    
    Terminal Output
    : Current GPU utilization in MB
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def split_str_list(str_list, bool_lower=False):
    str_list = str_list.strip("['']").replace('\n', '')
    list_list = str_list.split("' '")

    new_list = []
    if bool_lower:
        for i in list_list:
            new_list.append(i.lower())

        list_list = new_list

    return list_list

def mapping_labels(list_labels):
    list_mapped_labels = []

    for i in list_labels:
        list_mapped_labels.append(label2id[i])
    return list_mapped_labels

for j in ['train', 'valid']:
    # with open(os.path.join(path_dir, f'{j}.csv'), 'r') as f:
    #     list_lines = f.readlines()

    # list_text = []
    # list_toks = []
    # list_nertags = []

    # list_tok_cur = []
    # list_ner_cur = []

    # for l in list_lines:
    #     if l.strip() == "":
    #         list_toks.append(list_tok_cur)
    #         list_nertags.append(list_ner_cur)
            
    #         list_text.append(" ".join(list_tok_cur))
    #         list_tok_cur = []
    #         list_ner_cur = []
    #     else:
    #         try:
    #             text, ner_tags = l.strip().split(' ')
    #             list_tok_cur.append(text)
    #             list_ner_cur.append(label2id[ner_tags])
    #         except:
    #             list_tok_cur.append([])
    #             list_ner_cur.append([])

    # pl_data = pl.DataFrame({'text': list_text, 'tokens': list_toks, 'ner_tags': list_nertags}).to_pandas()
    # sentence,type,tokens,prefix_tokens,ner_tags,prefix_tags
    pl_data = pl.read_csv(os.path.join(path_dir, f'{j}.csv')).filter(pl.col('type')=='race').rename({'sentence': 'text'}).select(
        pl.col('text').str.to_lowercase(),
        pl.col('tokens').map_elements(lambda x: split_str_list(x), bool_lower=True),
        pl.col('ner_tags').map_elements(lambda x: split_str_list(x)),
    ).with_columns(
        pl.col('ner_tags').map_elements(lambda x: mapping_labels(x))
    )

    # pl_data = pl_data[:8000]

    pd_data = pl_data.to_pandas()
    # pd_data = pd.read_csv(os.path.join(path_dir, f'{j}.csv')).rename(columns={'sentence': 'text'})[['text', 'tokens', 'ner_tags']]
    c_dataset = Dataset.from_dict(pd_data)
    dict_to_be[j] = c_dataset

    # f.close()

electrical_ner_dataset = DatasetDict(dict_to_be)

print_gpu_utilization()
# electrical_ner_dataset = load_dataset(DATASET_ID, trust_remote_code=True)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)
print_gpu_utilization()
# print(type(electrical_ner_dataset))

def tokenize_and_align_labels(examples, label_all_tokens=True):
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

    Parameters:
    examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                     - "tokens": list of words in a sentence.
                     - "ner_tags": list of corresponding entity tags for each word.

    label_all_tokens (bool): A flag to indicate whether all tokens should have labels.
                             If False, only the first token of a word will have a label,
                             the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # print(word_ids)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `<s>` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_electrical_ner_dataset = electrical_ner_dataset.map(tokenize_and_align_labels, batched=True)
print_gpu_utilization()
del tokenizer
gc.collect()
print_gpu_utilization()
# ## prep

# label_list= tokenized_electrical_ner_dataset["train"].features["ner_tags"]
# print(label_list)

# label_list= tokenized_electrical_ner_dataset["train"].features["ner_tags"].feature.names
num_labels = 3
print_gpu_utilization()
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print_gpu_utilization()
args = TrainingArguments(
    output_dir= "ModernBERT-domain-classifier",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=10,
    bf16=True,
    optim="adamw_torch_fused",

    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="recall",
)

data_collator = DataCollatorForTokenClassification(tokenizer)
print_gpu_utilization()

def compute_metrics(eval_preds):

    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        [id2label[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [id2label[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
    ]
    metric = evaluate.load("seqeval")
    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_electrical_ner_dataset["train"],
    eval_dataset=tokenized_electrical_ner_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=(lambda logits, labels: torch.argmax(logits, dim=-1))
)
print_gpu_utilization()
trainer.train()

print_gpu_utilization()
results = pd.DataFrame(trainer.state.log_history)
results = results[['epoch', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']]
results.dropna(inplace=True)
results.reset_index(drop=True, inplace=True)

OUTPUT_MODEL = 'logs/testing'

# Saving evaluation results in a CSV format for easy visualization and comparison.
results.to_csv(f"{OUTPUT_MODEL}-results.csv", index=False)

model.save_pretrained(OUTPUT_MODEL)
tokenizer.save_pretrained(OUTPUT_MODEL)

# id2label = {
#     str(i): label for i,label in enumerate(label_list)
# }
# label2id = {
#     label: str(i) for i,label in enumerate(label_list)
# }

config = json.load(open(f"{OUTPUT_MODEL}/config.json"))

config["id2label"] = id2label
config["label2id"] = label2id

json.dump(config, open(f"{OUTPUT_MODEL}/config.json","w"))

print_gpu_utilization()
torch.cuda.empty_cache()
# accelerator.free_memory()
gc.collect()
print_gpu_utilization()