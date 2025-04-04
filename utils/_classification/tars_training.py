from flair.data import Corpus
from flair.datasets import TREC_6, CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# Load custom dataset
# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data'

# column format indicating which columns hold the text and label(s)
column_name_map = {4: "text", 1: "label_topic", 2: "label_subtopic"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter='\t',    # tab-separated files
)

# from flair.data import Corpus
# from flair.datasets import ClassificationCorpus

# # this is the folder in which train, test and dev files reside
# data_folder = '/path/to/data/folder'

# # load corpus containing training, test and dev data
# corpus: Corpus = ClassificationCorpus(data_folder,
#                                       test_file='test.txt',
#                                       dev_file='dev.txt',
#                                       train_file='train.txt',
#                                       label_type='topic',
#                                       )


# # this is the folder in which train, test and dev files reside
# data_folder = '/path/to/data/folder'

# # load corpus by pointing to folder. Train, dev and test gets identified automatically.
# corpus: Corpus = ClassificationCorpus(data_folder,
#                                       label_type='topic',
#                                       )

# 2. what label do we want to predict?
label_type = 'document_class'

# 3. create the label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

# 4. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(classifier, corpus)

# 7. run training with fine-tuning
trainer.fine_tune('resources/taggers/question-classification-with-transformer',
                  learning_rate=5.0e-5,
                  mini_batch_size=4,
                  max_epochs=10,
                  )