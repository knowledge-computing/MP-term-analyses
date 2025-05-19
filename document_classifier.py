import polars as pl
import pickle
from pynvml import *

from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus
from flair.datasets import ColumnCorpus



def print_gpu_utilization() -> None:
    """
    Print current GPU utilization stat
    
    Terminal Output
    : Current GPU utilization in MB
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def split_str_list(str_list, bool_lower=False):
    str_list = str_list.strip("['']").replace('\n', '')
    list_list = str_list.split(" ")

    new_list = []
    for i in list_list:
        new_list.append(int(i))

    list_list = new_list

    if (sum(list_list) == 0):
        return 'NOPE'

    return 'RACE'

data_folder = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/tars_data/csv'
column_name_map = {0: 'text', 1: 'label'}

# data_split = 'train'

# pl_data = pl.read_csv(f'/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/formatted/splitted/mn-dakota/{data_split}.csv').select(
#     pl.col('prefix_tags').map_elements(lambda x: split_str_list(x)),
#     pl.col('sentence').str.to_lowercase(),
# )

# pl_data = pl_data.write_csv(f'/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/tars_mini/{data_split}.txt', separator='\t')

# print(pl_data)

# 1. define label names in natural language since some datasets come with cryptic set of labels
label_name_map = {"RACE": "contain racial restriction",
                  "NOPE": "doesn't contain"}
# label_name_map = {'ENTY': 'question about entity',
#                   'DESC': 'question about description',
#                   'ABBR': 'question about abbreviation',
#                   'HUM': 'question about person',
#                   'NUM': 'question about number',
#                   'LOC': 'question about location'
#                   }

# corpus: Corpus = ColumnCorpus(data_folder, column_name_map,
#                               train_file='train.txt',
#                               test_file='test.txt',
#                               dev_file='dev.txt')

# corpus: Corpus = CSVClassificationCorpus(data_folder,
#                                          column_name_map=column_name_map,
#                                          skip_header=True,
#                                          delimiter=',',
#                                          label_type='prefix_tags',
# )

# # # 2. get the corpus
# # corpus: Corpus = TREC_6(label_name_map=label_name_map)

# # print(corpus)

# # 3. what label do you want to predict?
# label_type = 'prefix_tags'

# # 4. make a label dictionary
# label_dict = corpus.make_label_dictionary(label_type=label_type)

# # 5. start from our existing TARS base model for English
# tars = TARSClassifier.load("tars-large")

# # 5a: alternatively, comment out previous line and comment in next line to train a new TARS model from scratch instead
# # tars = TARSClassifier(embeddings="bert-base-uncased")

# # 6. switch to a new task (TARS can do multiple tasks so you must define one)
# tars.add_and_switch_to_new_task(task_name="question classification",
#                                 label_dictionary=label_dict,
#                                 label_type=label_type,
#                                 )

# # 7. initialize the text classifier trainer
# trainer = ModelTrainer(tars, corpus)

# # 8. start the training
# trainer.train(base_path='./tagger/restriction3',  # path to store the model artifacts
#               learning_rate=0.02,  # use very small learning rate
#               mini_batch_size=16,
#               mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
#               max_epochs=10,  # terminate after 10 epochs
#               main_evaluation_metric = ("macro avg", "f1-score"),
#               )


### Inference #####

# # 1. Load a pre-trained TARS model
tars = TARSClassifier.load('/home/yaoyi/pyo00005/Mapping_Prejudice/tagger/restriction/best-model.pt')

# 2. Check out what datasets it was trained on
# existing_tasks = tars.list_existing_tasks()
# print(f"Existing tasks are: {existing_tasks}")

# 3. Switch to a particular task that exists in the above list
tars.switch_to_task("question classification")

# 4. Prepare a test sentence

def get_prediction_label(str_sentence):
    sentence = Sentence(str_sentence)
    tars.predict(sentence)

    return str(sentence.get_label().value), float(sentence.get_label().score)

def tf_exist(list_tags):
    if "1" in list_tags:
        return 'RACE'
    else:
        return 'NOPE'

def compare_predict_true(prediction, truth):
    if (prediction == 'RACE') and (truth == 'RACE'):
        return 'TP'
    elif (prediction == 'NOPE') and (truth == 'RACE'):
        return 'FN'
    elif (prediction == 'RACE') and (truth == 'NOPE'):
        return 'FP'
    elif (prediction == 'NOPE') and (truth == 'NOPE'):
        return 'TN'

# pl_data = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/general_stereo/formatted/splitted/mn-dakota/test.csv').with_columns(
#     pl.col('prefix_tags').str.strip_chars('[]').str.split(' '),
#     predicted_label = pl.col('sentence').map_elements(lambda x: get_prediction_label(x))
# ).with_columns(
#     tmp = pl.col('prefix_tags').map_elements(lambda x: tf_exist(x))
# )

# with open('./testing.pkl', 'wb') as handle:
#     pickle.dump(pl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     # pl_data = pickle.load(handle)

# pl_data = pl_data.with_columns(
#     output_result = pl.struct(pl.all()).map_elements(lambda x: compare_predict_true(x['predicted_label'], x['tmp']))
# )
# print(pl_data)

sentence = Sentence("""no one other than of caucasian race can occupy this property""")
tars.predict(sentence)
print(sentence.get_label().value)


# print(f"TP: {pl_data.filter(pl.col('output_result') == 'TP').shape[0]}\nFN: {pl_data.filter(pl.col('output_result') == 'FN').shape[0]}\nTN: {pl_data.filter(pl.col('output_result') == 'TN').shape[0]}\nFP: {pl_data.filter(pl.col('output_result') == 'FP').shape[0]}")
# print(pl_data)