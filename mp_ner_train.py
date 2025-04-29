import os
from typing import Union, Dict, List, Tuple

import json
import math
import numpy as np
import polars as pl
import pandas as pd

from datasets import load_dataset
from transformers import (BertTokenizerFast, AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)
import evaluate

def split_data(df_data:Union[pl.DataFrame, pd.DataFrame],
               split_ratio:Dict[str, float],
               dir_output:str, class_col:str=None) -> None:
    """
    Splits data into train, valid, test sets. Follows split ratio. 
    If given class_col it will split with class consideration.
    """
    if isinstance(df_data, pd.DataFrame):
        df_data = pl.from_pandas(df_data)

    if class_col:
        list_dfs = df_data.partition_by(class_col)
    else:
        list_dfs = [df_data]

    list_splitted_dfs = []

    for d in list_dfs:
        len_data = d.shape[0]

        list_split = []
        for idx, (tag, ratio) in enumerate(split_ratio.items()):
            if idx == len(split_ratio) - 1:
                remaining_count = len_data - len(list_split)
                list_split.extend([tag] * remaining_count)
                break

            count = math.ceil(ratio * len_data)
            list_split.extend([tag] * count)

        d = d.sample(fraction=1, shuffle=True)
        d = d.with_columns(
            data_split = pl.Series(list_split)
        )
        list_splitted_dfs.append(d)

    df_data = pl.concat(
        list_splitted_dfs,
        how='vertical_relaxed'
    )

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    list_df_data = df_data.partition_by('data_split')
    for df in list_df_data:
        partition_name = df.item(0, 'data_split')
        df.write_csv(f'{os.path.join(dir_output, partition_name)}.csv')

def load_model_tokenizer(model_name:str, num_labels:int):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    return model, tokenizer, data_collator

def label_id_mapping():
    id2label = {
        str(i): label for i,label in enumerate(label_list)
    }
    label2id = {
        label: str(i) for i,label in enumerate(label_list)
    }

    return id2label, label2id

def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
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

def default(tokenized_data, output_dir):
    model, tokenizer, data_collator = load_model_tokenizer()

    global label_list
    # TODO: get label_list

    args = TrainingArguments(
        output_dir= output_dir,
        per_device_train_batch_size=32,
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
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    id2label, label2id = label_id_mapping()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = json.load(open(f"{output_dir}/config.json"))
    config["id2label"] = id2label
    config["label2id"] = label2id

    json.dump(config, open(f"{output_dir}/config.json","w"))

FILE_NAME = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/labeled_data/wi-milwaukee-county_deedpage_sample_post_zooniverse_wi-milwaukee-county_100pct_20250410_1716.csv'
dir_output = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/splitted_data/wi-milwaukee'
file_name = ''

dict_split = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
pl_data = pl.read_csv(FILE_NAME, infer_schema_length=0)
class_col = 'cov_confirmed'

split_data(df_data = pl_data, split_ratio=dict_split, dir_output=dir_output, class_col=class_col)