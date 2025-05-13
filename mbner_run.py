from transformers import pipeline
from transformers import BertTokenizerFast, DistilBertTokenizerFast 
from transformers import AutoModelForTokenClassification 

import spacy
nlp = spacy.load('en_core_web_sm') # Load the English Model

import os
import pandas as pd
import polars as pl
import pickle
import argparse

import warnings
from polars.exceptions import MapWithoutReturnDtypeWarning
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)

def clean_and_group_entities(ner_results, min_score=0.20):
    grouped_entities = []
    current_entity = None

    for result in ner_results:
        # Skip entities with score below threshold
        if result["score"] < min_score:
            if current_entity:
                # Add current entity if it meets threshold
                if current_entity["score"] >= min_score:
                    grouped_entities.append(current_entity['word'].strip())
                current_entity = None
            continue

        word = result["word"].replace("##", "")  # Remove subword token markers
        
        if current_entity and result["entity_group"] == current_entity["entity_group"] and result["start"] == current_entity["end"]:
            # Continue the current entity
            current_entity["word"] += word
            current_entity["end"] = result["end"]
            current_entity["score"] = min(current_entity["score"], result["score"])
            
            # If combined score drops below threshold, discard the entity
            if current_entity["score"] < min_score:
                current_entity = None
        else:
            # Finalize the current entity if it meets threshold
            if current_entity and current_entity["score"] >= min_score:
                grouped_entities.append(current_entity['word'].strip())
            
            # Start a new entity
            current_entity = {
                "entity_group": result["entity_group"],
                "word": word,
                "start": result["start"],
                "end": result["end"],
                "score": result["score"]
            }

    # Add the last entity if it meets threshold
    if current_entity and current_entity["score"] >= min_score:
        grouped_entities.append(current_entity['word'].strip())

    return grouped_entities

def returning(text:str):
    ner_results = nlp(text)
    cleaned_results = clean_and_group_entities(ner_results)

    return cleaned_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TA2 FuseMine-Mineral Site Linking')

    parser.add_argument('--output',
                        help="")
    
    args = parser.parse_args()

    
    # OUTPUT_MODEL = '/home/yaoyi/pyo00005/Mapping_Prejudice/_archive/_small/testing'
    OUTPUT_MODEL = f'{args.output}/testing'
    # OUTPUT_MODEL = '/home/yaoyi/pyo00005/Mapping_Prejudice/logs/regular/testing'

    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(OUTPUT_MODEL)
    tokenizer = DistilBertTokenizerFast.from_pretrained(OUTPUT_MODEL)

    nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

    # path_dir = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/mapped_to_org/mn-anoka'

    label2id = {'O':0, 'B-racial':1, 'I-racial':2}
    id2label = {0:'O', 1:'B-racial', 2:'I-racial'}

    dict_to_be = {}

    county_name = 'washington'
    # county_name = 'dakota'
    pl_data = pl.read_csv(f'/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/formatted/splitted/mn-{county_name}/test.csv')
    # pl_data = pl_data.filter(pl.col('tokens').list.len() > 0)

    # pl_data = pl_data[:10]
    pl_data = pl_data.with_columns(
        ner_identified = pl.col('sentence').map_elements(lambda x: returning(x))
    )

    with open(f'{args.output}/{county_name}.pkl', 'wb') as handle:
        pickle.dump(pl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)