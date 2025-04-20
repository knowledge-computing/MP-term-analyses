# import polars as pl
# import json
# import os
# from typing import Union, List, Dict
# from nltk import tokenize
# import math
# import warnings
# import nltk

# warnings.filterwarnings("ignore", category=pl.exceptions.MapWithoutReturnDtypeWarning)

# pl1 = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/mp_model/_seed_terms/mp_reduced.csv')['term'].to_list()
# pl2 = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/mp_model/_seed_terms/scaling-ai.csv')['term'].to_list()
# pl1.extend(pl2)

# new_list = []
# for p in pl1:
#     new_list.append(p.lower())

# new_list = list(set(new_list))

# def merge_basic_fuzzy(filtered_basic:Union[str, dict], 
#                       filtered_fuzzy:Union[str, dict]):
#     if not filtered_basic and not filtered_fuzzy:
#         return {}
    
#     if filtered_basic:
#         if isinstance(filtered_basic, str):
#             filtered_basic = json.loads(filtered_basic)
#             del filtered_basic['workflow']
#             del filtered_basic['lookup']
#             del filtered_basic['uuid']

#     if filtered_fuzzy:
#         if isinstance(filtered_fuzzy, str):
#             filtered_fuzzy = json.loads(filtered_fuzzy)
#             del filtered_fuzzy['workflow']
#             del filtered_fuzzy['lookup']
#             del filtered_fuzzy['uuid']

#         if filtered_basic and filtered_fuzzy:
#             filtered_basic.update(filtered_fuzzy)

#         else:
#             filtered_basic = filtered_fuzzy

#     filtered_dict = {}
#     for k, v in filtered_basic.items():
#         if k in new_list:
#             for i in v:
#                 try:
#                     filtered_dict[i].append(k)
#                 except:
#                     filtered_dict[i] = [k]

#     return filtered_dict

# def load_org_data(path_ocr_text:str,
#                   filtered_basic:Union[str, dict], 
#                   filtered_fuzzy:Union[str, dict]) -> str:
#     default_directory = '/home/yaoyi/pyo00005/Mapping_Prejudice/data'
    
#     filtered_dict = merge_basic_fuzzy(filtered_basic=filtered_basic,
#                                       filtered_fuzzy=filtered_fuzzy)

#     with open(os.path.join(default_directory, path_ocr_text), 'r') as f:
#         lines_text = f.readlines()

#     list_document = []
#     list_labels = []

#     dictionary_item_keys = list(filtered_dict.keys())

#     for current_idx, current_line in enumerate(lines_text):
#         list_tokens_org = tokenize.word_tokenize(current_line)
#         list_tokens = [x.lower() for x in list_tokens_org]
        
#         # list_tokens = tokenize.word_tokenize(current_line.lower())
#         list_er_labels = ['O'] * len(list_tokens)

#         if current_idx in dictionary_item_keys:
#             k = current_idx
#             v = filtered_dict[k]

#             for tok in v:
#                 for i, j in enumerate(list_tokens):
#                     if j in ['negros', 'caucasion', 'caucasien']:
#                         list_er_labels[i] = 'B-racial'

#                 if tok in list_tokens:
#                     list_matching_indexes = [i for i, j in enumerate(list_tokens) if (j == tok)]
#                     for alpha in list_matching_indexes:
#                         list_er_labels[alpha] = 'B-racial'

#                 else:
#                     tokenized_matches = tokenize.word_tokenize(tok.lower())
#                     if len(tokenized_matches) > 1:
#                         for idx, i in enumerate(list_tokens):
#                             if ((i == tokenized_matches[0])):
#                                 bool_true_stamp = False
#                                 for l in range(1, len(tokenized_matches)):
#                                     if ((list_tokens[idx + l] == tokenized_matches[l])):
#                                         bool_true_stamp = True
#                                     else:
#                                         bool_true_stamp = False
#                                 if bool_true_stamp:
#                                     list_er_labels[idx] = 'B-racial'
#                                     for l in range(1, len(tokenized_matches)):
#                                         list_er_labels[idx + l] = 'I-racial'

#         list_document.extend(list_tokens_org)
#         list_document.append(" ")

#         list_labels.extend(list_er_labels)
#         list_labels.append(" ")

#     return {'doc': list_document, 'label': list_labels}

# def split_tvt(pl_data:pl.DataFrame,
#               col_perclass:str=None):
    
#     list_pls = [pl_data]
#     if col_perclass:
#         list_pls = pl_data.partition_by(col_perclass)

#     list_spllited_pls = []
#     for df in list_pls:
#         df = df.sample(fraction=1, shuffle=True)    # Shuffling dataframe to induce random
        
#         tvt_split = []

#         len_df = df.shape[0]
#         train_size = math.ceil(len_df * 0.8)
#         valid_size = math.ceil(len_df * 0.1)
#         test_size = len_df - train_size - valid_size

#         tvt_split.extend([0] * train_size)
#         tvt_split.extend([1] * valid_size)
#         tvt_split.extend([2] * test_size)

#         df = df.with_columns(
#             tvt_split = pl.Series(tvt_split)
#         )
#         list_spllited_pls.append(df)

#     processed_pl = pl.concat(
#         list_spllited_pls,
#         how='vertical_relaxed'
#     )

#     return processed_pl

# county_detail = 'mn-washington'

# pl_data = pl.read_csv(f'/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/labeled_data/mn-dakota-county_deedpage_sample_post_zooniverse_mn-dakota-county_100pct_20250410_1714.csv',
#                       infer_schema_length=0)

# pl_data = pl_data.select(
#     pl.col(['page_ocr_text', 'cov_confirmed', 'hit_contents_basic', 'hit_contents_fuzzy']),
# ).drop_nulls('cov_confirmed').with_columns(
#     pl.when(pl.col('cov_confirmed') == False)
#     .then(None)
#     .otherwise(pl.col('hit_contents_basic')).alias('filtered_basic'),
#     pl.when(pl.col('cov_confirmed') == False)
#     .then(None)
#     .otherwise(pl.col('hit_contents_fuzzy')).alias('filtered_fuzzy')
# ).drop(['hit_contents_basic', 'hit_contents_fuzzy']).with_columns(
#     original_text = pl.struct(pl.all()).map_elements(lambda x: load_org_data(path_ocr_text=x['page_ocr_text'],
#                                                                                  filtered_basic=x['filtered_basic'],
#                                                                                  filtered_fuzzy=x['filtered_fuzzy']))
# ).unnest('original_text')

# pl_data = split_tvt(pl_data, 'cov_confirmed')

# list_tvt_splitted = pl_data.partition_by('tvt_split')

# save_directory = f'/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/mapped_to_org/{county_detail}'
# try:
#     os.mkdir(save_directory)
# except:
#     pass

# import csv
# for i in list_tvt_splitted:
#     match i.item(0, 'tvt_split'):
#         case 0:
#             list_tokens = [row["doc"] for row in i.iter_rows(named=True)]
#             list_labels = [row["label"] for row in i.iter_rows(named=True)]

#             with open(os.path.join(save_directory, 'train.txt'), 'w') as f:
#                 for idx, i in enumerate(list_tokens):
#                     for idxj, j in enumerate(i):
#                         f.write(f"{j} {list_labels[idx][idxj]}\n")

#                     f.write(f"\n")
#                 # writer = csv.writer(f, delimiter=' ')
#                 # writer.writerows(zip(i['doc'].to_list(), i['label'].to_list()))
                
#         case 1:
#             list_tokens = [row["doc"] for row in i.iter_rows(named=True)]
#             list_labels = [row["label"] for row in i.iter_rows(named=True)]

#             with open(os.path.join(save_directory, 'train.txt'), 'w') as f:
#                 for idx, i in enumerate(list_tokens):
#                     for idxj, j in enumerate(i):
#                         f.write(f"{j} {list_labels[idx][idxj]}\n")

#                     f.write(f"\n")
#         case 2: 
#             list_tokens = [row["doc"] for row in i.iter_rows(named=True)]
#             list_labels = [row["label"] for row in i.iter_rows(named=True)]

#             with open(os.path.join(save_directory, 'train.txt'), 'w') as f:
#                 for idx, i in enumerate(list_tokens):
#                     for idxj, j in enumerate(i):
#                         f.write(f"{j} {list_labels[idx][idxj]}\n")

#                     f.write(f"\n")

import flair
print(flair.__version__)

import torch
torch.cuda.empty_cache()

from flair.data import Corpus
from flair.datasets import ColumnCorpus
columns = {0 : 'text', 1 : 'ner'}

data_folder = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/mapped_to_org/mn-anoka'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='valid.txt')

tag_dictionary = corpus.make_label_dictionary(label_type='ner')
print(tag_dictionary)

from flair.embeddings import TransformerWordEmbeddings
embeddings = TransformerWordEmbeddings(
model='roberta-base',
layers='-1',
subtoken_pooling='first',
fine_tune=True,
use_context=True)

from flair.models import SequenceTagger
tagger = SequenceTagger(
hidden_size=256,
embeddings=embeddings,
tag_dictionary=tag_dictionary,
tag_type='ner')

from flair.trainers import ModelTrainer
trainer = ModelTrainer(tagger, corpus)
trainer.train(
'./models/flair_ner',
learning_rate=0.1,
mini_batch_size=8,
max_epochs=200)

# RUnning
# from flair.models import SequenceTagger
# tagger = SequenceTagger.load('models/flair_ner/final-model.pt')
# from flair.data import Sentence
# sentence = Sentence('sample sentence')
# tagger.predict(sentence)
# print(sentence)