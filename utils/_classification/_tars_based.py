from typing import List
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus

# testing
from flair.datasets import TREC_6

def tars_classifier_zero(list_sentences:List[str]|str,
                         classes:List[str]|str,
                         model:str='tars-base') -> List[str]|str:
                         
    tars = TARSClassifier.load(model)

    if isinstance(classes, str):
        classes = [classes]

    if isinstance(list_sentences, str):
        sentence = Sentence(list_sentences)
        tars.predict_zero_shot(sentence, classes)
        return sentence
    
    list_identified_sentences = []
    for line in list_sentences:
        sentence = Sentence(line)
        tars.predict_zero_shot(sentence, classes)
        list_identified_sentences.append(sentence)

    return list_identified_sentences

def t(label_dict, label_type, corpus):
    tars = TARSClassifier.load("tars-base")

    tars.add_and_switch_to_new_task(task_name="question classification",
                                label_dictionary=label_dict,
                                label_type=label_type,
                                )

    trainer = ModelTrainer(tars, corpus)

    trainer.train(base_path='resources/taggers/trec',  # path to store the model artifacts
              learning_rate=0.02,  # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
              max_epochs=5,  # terminate after 10 epochs
              )
    return 0

def tmp():
    label_name_map = {'ENTY': 'question about entity',
                  'DESC': 'question about description',
                  'ABBR': 'question about abbreviation',
                  'HUM': 'question about person',
                  'NUM': 'question about number',
                  'LOC': 'question about location'
                  }

    # 2. get the corpus
    corpus: Corpus = TREC_6(label_name_map=label_name_map)

    print(type(corpus.train))
    print(type(corpus))

    for i in list(range(10)):
        line = corpus.train[i]
        print(type(line), line)
        # print(line.labels)

    sentence = Sentence('The grass is green.')
    sentence.add_label('classification', 'question about description')
    print(type(sentence), sentence)