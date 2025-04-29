import argparse
from mpterm import MPTerm

def main(path_data:str,
         dir_output:str,file_output:str) -> None:
    
    mpterm = MPTerm(path_data=path_data,
                    dir_output=dir_output, file_output=file_output)

    # Check if path_data exists
    # If not exist throw error

    # Create dir_output folder

    # procmine = ProcMine(path_data=path_data, path_map=path_map,
    #                     dir_output=dir_output, file_output=file_output)
    
    # # Load data file, load map file, check output directory, check entities directory
    # procmine.prepare_data_paths()

    # # Process database
    # procmine.process()

    # Save output
    mpterm.save_output()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mapping Prejudice')

    parser.add_argument('--path_data', required=True,
                        help='Directory or file where the mineral site database is located')
    
    # parser.add_argument('--path_map',
    #                     help='CSV file with label mapping information.')
    
    parser.add_argument('--output_directory', default='./output',
                        help='Directory for processed mineral site database')
    
    parser.add_argument('--output_filename', required=True
                        help='Filename for processed mineral site database')
    
    args = parser.parse_args()

    main(path_data=args.path_data,
         dir_output=args.output_dir, file_output=args.output_file)