import os
import polars as pl
from transformers import BertTokenizerFast

path_dir = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/mapped_to_org/mn-anoka'

from datasets import load_dataset, Dataset, DatasetDict

# MODEL_ID = "answerdotai/ModernBERT-large"

# tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)

# def tokenize_and_align_labels(examples, label_all_tokens=True):
#     """
#     Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
#     Named Entity Recognition (NER) tasks where alignment of the labels is necessary after tokenization.

#     Parameters:
#     examples (dict): A dictionary containing the tokens and the corresponding NER tags.
#                      - "tokens": list of words in a sentence.
#                      - "ner_tags": list of corresponding entity tags for each word.

#     label_all_tokens (bool): A flag to indicate whether all tokens should have labels.
#                              If False, only the first token of a word will have a label,
#                              the other tokens (subwords) corresponding to the same word will be assigned -100.

#     Returns:
#     tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
#     """
#     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
#     labels = []
#     for i, label in enumerate(examples["ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         print(word_ids)
#         # word_ids() => Return a list mapping the tokens
#         # to their actual word in the initial sentence.
#         # It Returns a list indicating the word corresponding to each token.
#         previous_word_idx = None
#         label_ids = []
#         # Special tokens like `<s>` and `<\s>` are originally mapped to None
#         # We need to set the label to -100 so they are automatically ignored in the loss function.
#         for word_idx in word_ids:
#             if word_idx is None:
#                 # set –100 as the label for these special tokens
#                 label_ids.append(-100)
#             # For the other tokens in a word, we set the label to either the current label or -100, depending on
#             # the label_all_tokens flag.
#             elif word_idx != previous_word_idx:
#                 # if current word_idx is != prev then its the most regular case
#                 # and add the corresponding token
#                 label_ids.append(label[word_idx])
#             else:
#                 # to take care of sub-words which have the same word_idx
#                 # set -100 as well for them, but only if label_all_tokens == False
#                 label_ids.append(label[word_idx] if label_all_tokens else -100)
#                 # mask the subword representations after the first subword

#             previous_word_idx = word_idx
#         labels.append(label_ids)
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

# label2id = {'O':0, 'B-racial':1, 'I-racial':2}
# id2label = {0:'O', 1:'B-racial', 2:'I-racial'}

# for j in ['train', 'valid', 'text']:

#     with open(os.path.join(path_dir, f'{j}.txt'), 'r') as f:
#         list_lines = f.readlines()

#     list_text = []
#     list_toks = []
#     list_nertags = []

#     list_tok_cur = []
#     list_ner_cur = []

#     for l in list_lines:
#         if l.strip() == "":
#             list_toks.append(list_tok_cur)
#             list_nertags.append(list_ner_cur)
            
#             list_text.append(" ".join(list_tok_cur))
#             list_tok_cur = []
#             list_ner_cur = []
#         else:
#             text, ner_tags = l.strip().split(' ')
#             list_tok_cur.append(text)
#             list_ner_cur.append(label2id[ner_tags])

#     pl_data = pl.DataFrame({'text': list_text, 'tokens': list_toks, 'ner_tags': list_nertags}).filter(pl.col('text') == ' ').to_pandas()
#     c_dataset = Dataset.from_dict(pl_data)
#     my_dataset_dict = DatasetDict({f:c_dataset})

# # tokenized_electrical_ner_dataset = my_dataset_dict.map(tokenize_and_align_labels, batched=True)
# # print(tokenized_electrical_ner_dataset['train'].features["ner_tags"].feature.names)

# # def create_dataset():
# #     dataset = load_dataset('csv', data_files={'train': 'train.txt', 'validation': 'val.txt', 'test': 'test.txt'}, sep=",", 
# #                                 names=["text", "tokens", "ner_tags"])
# #     labels = df['label'].unique().tolist()
# #     ClassLabels = ClassLabel(num_classes=len(labels), names=labels)
    

# #     dataset.class_encode_column('ner_tags')

# #     return 0
path_dir = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/mapped_to_org/mn-anoka'

label2id = {'O':0, 'B-racial':1, 'I-racial':2}
id2label = {0:'O', 1:'B-racial', 2:'I-racial'}

dict_to_be = {}

for j in ['train', 'valid', 'test']:
    with open(os.path.join(path_dir, f'{j}.txt'), 'r') as f:
        list_lines = f.readlines()

    list_text = []
    list_toks = []
    list_nertags = []

    list_tok_cur = []
    list_ner_cur = []

    for l in list_lines:
        if l.strip() == "":
            list_toks.append(list_tok_cur)
            list_nertags.append(list_ner_cur)
            
            list_text.append(" ".join(list_tok_cur))
            list_tok_cur = []
            list_ner_cur = []
        else:
            try:
                text, ner_tags = l.strip().split(' ')
                list_tok_cur.append(text)
                list_ner_cur.append(label2id[ner_tags])
            except:
                list_tok_cur.append([])
                list_ner_cur.append([])

    pl_data = pl.DataFrame({'text': list_text, 'tokens': list_toks, 'ner_tags': list_nertags}).to_pandas()
    c_dataset = Dataset.from_dict(pl_data)
    dict_to_be[j] = c_dataset

    f.close()

electrical_ner_dataset = DatasetDict(dict_to_be)

print(electrical_ner_dataset)