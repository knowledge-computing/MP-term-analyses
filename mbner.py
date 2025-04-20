import os 
from transformers import BertTokenizerFast
from datasets import load_dataset

DATASET_ID = "disham993/ElectricalNER"
MODEL_ID = "answerdotai/ModernBERT-large"

electrical_ner_dataset = load_dataset(DATASET_ID, trust_remote_code=True)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)

print(type(electrical_ner_dataset))

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
        print(word_ids)
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

# ## prep

label_list= tokenized_electrical_ner_dataset["train"].features["ner_tags"]
print(label_list)

label_list= tokenized_electrical_ner_dataset["train"].features["ner_tags"].feature.names
# num_labels = len(label_list)

# model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, num_labels=num_labels)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# args = TrainingArguments(
#     output_dir= "ModernBERT-domain-classifier",
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=16,
#     learning_rate=5e-5,
#     num_train_epochs=10,
#     bf16=True, # bfloat16 training 
#     optim="adamw_torch_fused", # improved optimizer 
#     # logging & evaluation strategies
#     logging_strategy="steps",
#     logging_steps=100,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
# )

# data_collator = DataCollatorForTokenClassification(tokenizer)

# def compute_metrics(eval_preds):
#     """
#     Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
#     The function computes precision, recall, F1 score and accuracy.

#     Parameters:
#     eval_preds (tuple): A tuple containing the predicted logits and the true labels.

#     Returns:
#     A dictionary containing the precision, recall, F1 score and accuracy.
#     """
#     pred_logits, labels = eval_preds

#     pred_logits = np.argmax(pred_logits, axis=2)
#     # the logits and the probabilities are in the same order,
#     # so we don’t need to apply the softmax

#     # We remove all the values where the label is -100
#     predictions = [
#         [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(pred_logits, labels)
#     ]

#     true_labels = [
#       [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
#        for prediction, label in zip(pred_logits, labels)
#     ]
#     metric = evaluate.load("seqeval")
#     results = metric.compute(predictions=predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

# trainer = Trainer(
#     model,
#     args,
#     train_dataset=tokenized_electrical_ner_dataset["train"],
#     eval_dataset=tokenized_electrical_ner_dataset["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer.train()


# results = pd.DataFrame(trainer.state.log_history)
# results = results[['epoch', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']]
# results.dropna(inplace=True)
# results.reset_index(drop=True, inplace=True)

# OUTPUT_MODEL = 'logs/testing'

# # Saving evaluation results in a CSV format for easy visualization and comparison.
# results.to_csv(f"{OUTPUT_MODEL}-results.csv", index=False)

# model.save_pretrained(OUTPUT_MODEL)
# tokenizer.save_pretrained(OUTPUT_MODEL)

# id2label = {
#     str(i): label for i,label in enumerate(label_list)
# }
# label2id = {
#     label: str(i) for i,label in enumerate(label_list)
# }

# config = json.load(open(f"{OUTPUT_MODEL}/config.json"))

# config["id2label"] = id2label
# config["label2id"] = label2id

# json.dump(config, open(f"{OUTPUT_MODEL}/config.json","w"))