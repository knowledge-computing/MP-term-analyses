import argparse
from typing import Union
from mpterm import MPTerm

def main(path_data:str=None, json_input:Union[str, dict]=None,
         dir_output:str=None, file_output:str=None,
         bool_local:bool=False):
    
    # if bool_local:
    #     mpterm = MPTerm(input_info=path_data,
    #                     dir_output=dir_output, file_output=file_output)
    # else:
    #     mpterm = MPTerm(input_info=json_input,
    #                     dir_output=dir_output, file_output=file_output)

    dummy_json = {
        "statusCode": 200,
        "body": {
            "message": "hello world",
            "bucket": "covenants-deed-images",
            "orig_img": "raw/test-county/Deeds/19499142059800.001.tif",
            "ocr_json": "ocr/json/mn-anoka-county/9/30677860.json",
            "web_img": "web/test-county/Deeds/ceb01a595e884949a3d06edd6ccdfffc.jpg",
            "uuid": "ceb01a595e884949a3d06edd6ccdfffc"
        }
    }

    mpterm = MPTerm(input_info=dummy_json,
                    dir_output=dir_output, file_output=file_output)

    # Load data to variables
    mpterm.load_data(bool_local=bool_local)

    # Run document classification
    mpterm.doc_classify()

    # Print classification output
    print(mpterm.return_output(pipeline='dc'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mapping Prejudice (document classification)')

    parser.add_argument('--local', default=False,
                        help='Indication if running on local', action='store_true')

    parser.add_argument('--path_data', default=None,
                        help='Path to the deed document')
    
    parser.add_argument('--json_input', default=None,
                        help='String format JSON input')
    
    parser.add_argument('--output_dir', default='./output',
                        help='Directory of output')
    
    parser.add_argument('--output_file', default=None,
                        help='Filename of output')
    
    args = parser.parse_args()

    main(path_data=args.path_data, json_input=args.json_input,
         dir_output=args.output_dir, file_output=args.output_file,
         bool_local=args.local)