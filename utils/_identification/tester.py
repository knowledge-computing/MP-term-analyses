# import spacy

# nlp = spacy.load('en_core_web_lg') 
  
# sentence = "Apple is looking at buying U.K. startup for $1 billion"
  
# doc = nlp(sentence) 
  
# for ent in doc.ents: 
#     print(ent.text, ent.start_char, ent.end_char, ent.label_) 

from datasets import load_dataset

wnut = load_dataset("wnut_17")
label_list = wnut["train"].features[f"ner_tags"].feature.names

print(wnut["train"][0])

print(label_list)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)

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
print(tokenized_wnut)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

labels = [label_list[i] for i in example[f"ner_tags"]]


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

id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
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

# model_name = "/home/yaoyi/pyo00005/Mapping_Prejudice/utils/_data/models"
# trainer.train()
# trainer.save_model(model_name)

text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

from transformers import pipeline

classifier = pipeline("ner", model="/home/yaoyi/pyo00005/Mapping_Prejudice/utils/_data/models/my_awesome_wnut_model")
print(classifier(text))

# import torch
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# inputs = tokenizer(text, return_tensors="pt")

# from transformers import AutoModelForTokenClassification

# model = AutoModelForTokenClassification.from_pretrained(model_name)
# with torch.no_grad():
#     logits = model(**inputs).logits

# predictions = torch.argmax(logits, dim=2)
# predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
# predicted_token_class