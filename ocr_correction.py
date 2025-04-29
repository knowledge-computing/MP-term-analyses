from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BartTokenizerFast
import spacy
nlp = spacy.load('en_core_web_sm') # Load the English Model

model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-large-ocr')
tokenizer = AutoTokenizer.from_pretrained('pykale/bart-large-ocr')
fasttokenizer = BartTokenizerFast.from_pretrained('pykale/bart-large-ocr')
generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024)

def ocr_correction(list_sentences:List[str]):
    # tokenized_org = []
    # tokenized_overlap = []
    # for idx, i in enumerate(list_sentences):
    #     # if idx != 0: 
    #     #     tmp_sentence = 
    #     # tokenized_inputs = fasttokenizer(i, truncation=True)
    #     pred = generator(i)[0]['generated_text']
    #     tokenized_org.append(pred)

    # print(tokenized_org)
    # print(tokenized_overlap)
    document_block = "".join(list_sentences)
    document_block = document_block.replace("\n", " ")


    # print(document_block)
    # print(document_block)

    ocr = "bred Dollars ( $ 15 00 ) exclusive of decerating no mearo on person of african des cent no chinese or g panese."
    pred = generator(ocr)[0]['generated_text']
    print(pred)

    return document_block
    # return 0

def sentence_tokenize(text:str) -> List[str]:
    doc = nlp(text)
    list_newsentence = []
    for sent in doc.sents:
        list_newsentence.append(sent)
        print(sent)
        print(generator(str(sent))[0]['generated_text'])

    return list_newsentence

with open('/home/yaoyi/pyo00005/Mapping_Prejudice/data/ocr/txt/mn-anoka-county/1/26360876_SPLITPAGE_1.txt', 'r') as f:
    list_lines = f.readlines()

doc_block = ocr_correction(list_lines)
sentence_tokenize(doc_block)

# print(list_lines)