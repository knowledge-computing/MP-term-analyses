# Mapping Prejudice: Racial Term Identification from Historical Deed Documents
This repository contains the Mapping Prejudice project pipeline for identifying racially restrictive language in historical deed documents. It provides two core features:
- **Entity Identification**: Uses [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) to recognize and extract racially restrictive terms and phrases based on their context within the document
- **Document Classification**: Uses [BERT-base](https://huggingface.co/google-bert/bert-base-uncased) with [TARS](https://aclanthology.org/2020.coling-main.285/) framework to determine whether any sentence in the document contains racially restrictive language.

## Installation:
### With `pip`
Install the required dependencies:
```
pip install -r requirements.txt
```

### With Poetry (Recommended)
[Poetry](https://python-poetry.org/) is recommended for better dependency and environment management. To install Poetry and install dependencies:
```
pip install -r poet.txt
poetry install --no-root
```

## Usage:
The pipeline supports two types of input: a JSON-formatted string or local directory paths. File input from S3 buckets is also supported via [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html#).

### Entity Identification
This module extracts racially restrictive terms from deed text using contextual entity recognition.

#### JSON Input
```
# pip
python3 run_identification.py --json_input <input_json>

# Poetry
poetry run python3 run_identification.py --json_input <input_json>
```

#### Local Data
```
# pip
python3 run_identification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>

# Poetry
poetry run python3 run_identification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>
```

### Document Classification
This module classifies whether any sentence in a document contains racially restrictive language.

#### JSON Input
```
# pip
python3 run_classification.py --json_input <input_json>

# Poetry
poetry run python3 run_classification.py --json_input <input_json>
```

#### Local Data
```
# pip
python3 run_classification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>

# Poetry
poetry run python3 run_classification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>
```