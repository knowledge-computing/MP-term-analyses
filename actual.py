import os
import time
from typing import List, Dict, Tuple, Union

from tqdm import tqdm
from tabulate import tabulate
from pynvml import *

import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from datasets import load_dataset, Dataset, DatasetDict
import transformers
from transformers import AutoTokenizer,  AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

import math
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_gpu_utilization() -> None:
    """
    Print current GPU utilization stat

    Parameters
    : None
    
    Return
    : None

    Terminal Output
    : Current GPU utilization in MB
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def preprocess_gen(tokenizer,
                   example:Dict[str, str]):
    
    return tokenizer(example['text'], truncation=True)

def preprocess_ner(tokenizer,
                   example:Dict[str, str]):
    
    encodings = tokenizer(example['text'], truncation=True, padding='max_length', is_split_into_words=True)
    labels = example['labels'] + [0] * (tokenizer.model_max_length - len(example['labels']))

    return { **encodings, 'labels': labels }

    return 0

# def split_data(df_data: pl.DataFrame,
#                split_ratio: Dict[str, float]={'train':0.8, 'valid':0.1, 'test':0.1}):
    
#     total_data_len = df_data.shape[0]
#     for k, v in split_ratio.items():
#         total_data_len
#     return 0

def df_to_tok_data(df_data:Union[pl.DataFrame, datasets.DataSets, pd.DataFrame],
                  tokenizer:transformers.Tokenizer,
                  task:str='ner'):

    if isinstance(df_data, pl.DataFrame) or isinstance(df_data, pd.DataFrame):
        if isinstance(df_data, pl.DataFrame):
            df_data = df_data.to_pandas()
            
        data_train = df_data[df_data['split'] == 'train']
        data_valid = df_data[df_data['split'] == 'valid']
        data_test = df_data[df_data['split'] == 'test']

        dd_input = DatasetDict()
        dd_input['train'] = Dataset.from_pandas(data_train)
        dd_input['valid'] = Dataset.from_pandas(data_valid)
        dd_input['test'] = Dataset.from_pandas(data_test)
    else:
        dd_input = df_data

    if task == 'ner':
        tokenized_input = dd_input.map(lambda x: preprocess_ner(x, tokenizer), batched=True)
    else:
        tokenized_input = dd_input.map(lambda x: preprocess_gen(x, tokenizer), batched=True)

    return tokenized_input

def create_label_dicts(list_labels:List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label dictionary

    Parameters
    : list_labels

    Return
    : id2label: 
    : label2id: 
    """
    id2label = {i: label.strip() for i, label in enumerate(list_labels)}
    label2id = {label.strip(): str(i) for i, label in id2label.items()}

    return label2id, id2label

def load_data(dataset_name:str='conll2003'):
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    return dataset

def load_tokenizer(model_name:str='answerdotai/ModernBERT-base'):
    tokenizer = AutoTokenizer(model_name)

    return tokenizer

def load_model(id2label:Dict[int, str],
               label2id:Dict[str, int],
               model_name:str='answerdotai/ModernBERT-base',
               num_labels:int=0):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.id2label = id2label
    model.config.label2id = label2id

    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
        labels, predictions, labels=labels, pos_label=1, average="weighted")
    
    return {"f1": float(score) if score == 1 else score}

def train_model(model, tokenized_input,
                output_dir:str='./outputs',
                per_device_train_batch:int=32,
                per_device_eval_batch:int=16,
                learning_rate:float=5e-5,
                num_train_epochs:int=5,
                bool_bf16:bool=True,
                optim:str='adamW_torch_fused',
                metric_for_best_model='f1'):

    print_gpu_utilization()

    training_args = TrainingArguments(
        output_dir= output_dir,
        per_device_train_batch_size=per_device_train_batch,
        per_device_eval_batch_size=per_device_eval_batch,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=bool_bf16,
        optim=optim,

        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,

    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_input["train"],
        eval_dataset=tokenized_input["valid"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print_gpu_utilization()

def run_model(model:str, tokenized_input,
              id2label:dict,
              trainer=None):
    print_gpu_utilization()

    if not trainer:
        # Load trainer from model
        pass

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    predictions, label_ids, metrics = trainer.predict(tokenized_input['test'])
    predicted_label = np.argmax(predictions, axis=1)

    print_gpu_utilization()
    return 0

def test_discriminative(tokenized_author_data, trainer, id2label:dict, bool_testing:bool):
    """
    Tests the model on the t
    Reference: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.predict
    
    : param: tokenized_author_data = 
    : param: trainer = 
    : param: id2label = dictionary consisting of the numeric id to author label mapping
    : param: bool_testing = boolean value indicating whether a testing file was provided (True) or not (False)
    """

    predictions, label_ids, metrics = trainer.predict(tokenized_author_data['test'])
    predicted_label = np.argmax(predictions, axis=1)
    test_text = tokenized_author_data['test']['text']

    if not bool_testing:
        test_label_ground_truth = tokenized_author_data['test']['label']
        
        compiled_data = pl.DataFrame(
            {
                "text":test_text,
                "author":test_label_ground_truth,
                "prediction":predicted_label
            }
        ).with_columns(
            pl.col('author').replace(id2label),
            pl.col('prediction').replace(id2label)
        )

        compiled_data.write_csv('./outputs/prediction_discriminative.csv',
                                separator=',')
        
        pl_incorrect_prediction = compiled_data.filter(
            pl.col('author') != pl.col('prediction')
        ).select(
            pl.col(['text', 'author', 'prediction'])
        )  
        pl_incorrect_prediction.write_csv('./outputs/incorrect_prediction_discriminative.csv',
                                          separator=',')
        
        # Calculates the accuracy of the language model on the dev set
        print("Results on dev set:")

        list_test_data = compiled_data.partition_by(
            'author'
        )

        table = []
        for i in list_test_data:
            author_name = i.item(0, 'author')

            total_count = i.shape[0]
            correct_count = i.filter(
                pl.col('prediction') == author_name
            ).shape[0]

            table.append([author_name, (correct_count/total_count) * 100])

            print(f"{author_name}\t\t{(correct_count/total_count) * 100:.1f}% correct")

        headers = ['Author', 'Accuracy']
        with open('./outputs/devset_result_discriminative.txt', 'w') as f:
            f.write(tabulate(table, headers, floatfmt='.1f'))

    else:
        # Print out in the format of what the author is line by line
        compiled_data = pl.DataFrame(
            {
                "text":test_text,
                "prediction":predicted_label
            }
        ).with_columns(
            pl.col('prediction').replace(id2label)
        )

        compiled_data.write_csv('./outputs/prediction_discriminative.csv',
                                separator=',')
        
        predicted_label = compiled_data['prediction'].to_list()

        for i in predicted_label:
            print(i)

    logging.info(f'Testing ended - Run Time: {time.time() - start_time}')


# num_labels = dataset['train'].features['ner_tags'].feature.num_classes

# dataset = dataset.map(add_encodings)
# dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# labels = dataset['train'].features['ner_tags'].feature
# label2id = { k: labels.str2int(k) for k in labels.names }
# id2label = { v: k for k, v in label2id.items() }

# # print(dataset['train'][0]

# model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=num_labels)
# # assign the 'id2label' and 'label2id' model configs
# model.config.id2label = id2label
# model.config.label2id = label2id

# # set the model in 'train' mode and send it to the device
# model.train().to(device)
# # initialize the Adam optimizer (used for training/updating the model)
# optimizer = optim.AdamW(params=model.parameters(), lr=1e-5)

# # set the number of epochs 
# n_epochs = 3
# # batch the train data so that each batch contains 4 examples (using 'batch_size')
# train_data = torch.utils.data.DataLoader(dataset['train'], batch_size=4)

# train_loss = []
# # iterate through the data 'n_epochs' times
# for epoch in tqdm(range(n_epochs)):
#     current_loss = 0
#     # iterate through each batch of the train data
#     for i, batch in enumerate(tqdm(train_data)):
#         # move the batch tensors to the same device as the 
#         batch = { k: v.to(device) for k, v in batch.items() }
#         # send 'input_ids', 'attention_mask' and 'labels' to the model
#         outputs = model(**batch)
#         # the outputs are of shape (loss, logits)
#         loss = outputs[0]
#         # with the .backward method it calculates all 
#         # of  the gradients used for autograd
#         loss.backward()
#         # NOTE: if we append `loss` (a tensor) we will force the GPU to save
#         # the loss into its memory, potentially filling it up. To avoid this
#         # we rather store its float value, which can be accessed through the
#         # `.item` method
#         current_loss += loss.item()
#         if i % 8 == 0 and i > 0:
#             # update the model using the optimizer
#             optimizer.step()
#             # once we update the model we set the gradients to zero
#             optimizer.zero_grad()
#             # store the loss value for visualization
#             train_loss.append(current_loss / 32)
#             current_loss = 0
#     # update the model one last time for this epoch
#     optimizer.step()
#     optimizer.zero_grad()

# fig, ax = plt.subplots(figsize=(10, 4))
# # visualize the loss values
# ax.plot(train_loss)
# # set the labels
# ax.set_ylabel('Loss')
# ax.set_xlabel('Iterations (32 examples)')
# fig.tight_layout()
# plt.show()

# model = model.eval()
# # batch the train data so that each batch contains 4 examples (using 'batch_size')
# test_data = torch.utils.data.DataLoader(dataset['test'], batch_size=4)

# # create the confusion matrix
# confusion = torch.zeros(num_labels, num_labels)

# # iterate through each batch of the train data
# for i, batch in enumerate(tqdm(test_data)):
#     # do not calculate the gradients
#     with torch.no_grad():
#         # move the batch tensors to the same device as the model
#         batch = { k: v.to(device) for k, v in batch.items() }
#         # send 'input_ids', 'attention_mask' and 'labels' to the model
#         outputs = model(**batch)
            
#     # get the sentence lengths
#     s_lengths = batch['attention_mask'].sum(dim=1)
#     # iterate through the examples
#     for idx, length in enumerate(s_lengths):
#         # get the true values
#         true_values = batch['labels'][idx][:length]
#         # get the predicted values
#         pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]
#         # go through all true and predicted values and store them in the confusion matrix
#         for true, pred in zip(true_values, pred_values):
#             confusion[true.item()][pred.item()] += 1

# # Normalize by dividing every row by its sum
# for i in range(num_labels):
#     confusion[i] = confusion[i] / confusion[i].sum()

# fig, ax = plt.subplots(figsize=(10, 10))
# # visualize the loss values
# ax.matshow(confusion.numpy())

# # get the labels
# labels = list(label2id.keys())
# ids = np.arange(len(labels))

# ax.set_ylabel('True Labels', fontsize='x-large')
# ax.set_xlabel('Pred Labels', fontsize='x-large')

# # set the x ticks
# ax.set_xticks(ids)
# ax.set_xticklabels(labels)

# # set the y ticks
# ax.set_yticks(ids)
# ax.set_yticklabels(labels)

# # plot figure
# fig.tight_layout()
# plt.show()