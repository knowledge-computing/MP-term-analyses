import argparse
from mpterm import MPTerm

def main(path_data:str,
         dir_output:str,file_output:str) -> None:
    
    mpterm = MPTerm(path_data=path_data,
                    dir_output=dir_output, file_output=file_output)

    # Save output
    mpterm.save_output()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mapping Prejudice')

    parser.add_argument('--path_data', required=True,
                        help='Directory or file where the mineral site database is located')
    
    parser.add_argument('--output_dir', default='./output',
                        help='Directory of output')
    
    parser.add_argument('--output_filename', required=True
                        help='Filename of output')
    
    args = parser.parse_args()

    main(path_data=args.path_data,
         dir_output=args.output_dir, file_output=args.output_file)