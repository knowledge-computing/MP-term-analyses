import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.tokenize import sent_tokenize

model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-base-ocr')
tokenizer = AutoTokenizer.from_pretrained('pykale/bart-base-ocr')
generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024)

# sample_data_path = '/home/yaoyi/pyo00005/Mapping_Prejudice/data/ocr/txt/mn-anoka-county/9/30634946_SPLITPAGE_2.txt'

# with open(sample_data_path, 'r') as f:
#     ocr = f.read()

# # ocr = "The defendant wits'fined �5 and costs."
# pred = generator(ocr)[0]['generated_text']
# print(pred)

def correct_ocr(text:str) -> str:
    try:
        output = generator(text)[0]['generated_text']
    except:
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, truncation=True, return_tensors='pt')
        output = model(**encoded_input)

    return output

def main(path_num:int):
    data_path = os.path.join('/home/yaoyi/pyo00005/Mapping_Prejudice/data/ocr/txt/mn-anoka-county', str(path_num))
    output_path = os.path.join('/home/yaoyi/pyo00005/Mapping_Prejudice/cleaned_data', str(path_num))

    for i in os.listdir(data_path):
        with open(os.path.join(data_path, i), 'r') as f:
            content = f.read()
        f.close()

        content = content.replace('\n', '')
        list_lines = sent_tokenize(content)

        
        with open(os.path.join(output_path, i), 'w') as f:
            for l in list_lines:
                f.write(correct_ocr(l))
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mapping Prejudice Project, Racial Covenant Identification Pipeline')

    parser.add_argument('--path_num', type=int,
                        help="Location of input file")
    # parser.add_argument('--dir_output', type=str,
    #                     help="Location of input file")

    args = parser.parse_args()

    main(args.path_num)