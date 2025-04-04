import polars as pl
import pandas as pd
import numpy as np

import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

# Delete this later
from datasets import load_dataset

def create_dataset_dictionary(pl_data, train_valid_test_map=None):
    list_split_types = list(set(pl_data['split_indicator'].to_list()))
    if not train_valid_test_map:
        train_valid_test_map = dict(zip(list_split_types, list_split_types))

    # Creating a dataset dictionary from the input polars dataframe that is splitted into train, validation, and test set
    # Reference: https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322/4 Second answer by akomma on February 22nd
    dataset_dictionary = DatasetDict()

    for split_type in list_split_types:
        split_tag = train_valid_test_map[split_type]

        pl_split = pl_data.filter(
            pl.col('split_indicator') == split_type
        ).drop('split_indicator')
        pd_split = pl_split.to_pandas()

        dataset_dictionary[split_tag] = Dataset.from_pandas(pd_split)

    return dataset_dictionary


wnut = load_dataset("wnut_17")
label_list = wnut["train"].features[f"ner_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

labels = [label_list[i] for i in example[f"ner_tags"]]

id2label = {
    0: "O",
    1: "B-covenant",
    2: "I-covenant",
}

label2id = {
    "O": 0,
    "B-covenant": 1,
    "I-covenant": 2,
}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

classifier = pipeline("ner", model="/home/yaoyi/pyo00005/Mapping_Prejudice/utils/_data/models/my_awesome_wnut_model")
print(classifier(text))