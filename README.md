# Mapping Prejudice: Racial Term Identification from Historical Deed Documents
This repository contains the Mapping Prejudice project pipeline for identifying racially restrictive language in historical deed documents. The model leverages [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) for entity recognition based on the contextual information within each document.

## Installation:
### With `pip`
Install the required dependencies:
```
pip install -r requirements.txt
```

### With Poetry (Recommended)
[Poetry](https://python-poetry.org/) is recommended for better dependency and envrionemtn management. To install Poetry and install dependencies:
```
pip install -r poet.txt
poetry install
```

## Usage:
The pipeline accepts two types of input: a JSON input string or data from a local directory. It also supports accessing files from an S3 bucket using [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html#).

### JSON Input
```
# If installed with pip
python3 run_identification.py --json_input <input_json>

# If using Poetry
poetry run python3 run_identification.py --json_input <input_json>
```

### Local Data
```
# If installed with pip
python3 run_identification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>

# If using Poetry
poetry run python3 run_identification.py --local --path_data <path_to_local_data> --output_dir <output_directory> --output_file <output_file_name>
```
