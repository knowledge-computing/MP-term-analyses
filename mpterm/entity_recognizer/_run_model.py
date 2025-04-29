from transformers import pipeline
from transformers import BertTokenizerFast, DistilBertTokenizerFast 
from transformers import AutoModelForTokenClassification 

def load_model(model_path:str):
    """

    Parameters
    : model_path (str): absolute path to the trained model

    Return
    : pipeline (obj):
    """
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    ner_pipeline = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

    return ner_pipeline

# model_fine_tuned = AutoModelForTokenClassification.from_pretrained(OUTPUT_MODEL)
#     tokenizer = DistilBertTokenizerFast.from_pretrained(OUTPUT_MODEL)

#     nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, aggregation_strategy="simple", device=0)