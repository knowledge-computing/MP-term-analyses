from Typing import Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def ocr_bart_basic(data: Dict[str, str]) -> Dict[str, str]:
    """
    
    """
    model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-base-ocr')
    tokenizer = AutoTokenizer.from_pretrained('pykale/bart-base-ocr')
    generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024)

    dict_ocr_corrected = {}

    for filename, text in data.items():
        pred = generator(text)[0]['generated_text']
        dict_ocr_corrected[filename] = pred

    return dict_ocr_corrected