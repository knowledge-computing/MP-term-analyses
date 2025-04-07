
import argparse
from mp_model import MpModel

def main(identification_method:str,
         bool_dev:bool=False) -> None:
    # Initialize mapping prejudice model
    mpmodel = MpModel()

    # Setup data

    # Run identification (focus: recall)
    
    # Run classification (focus: precision)

    if bool_dev:
        # Run evaluation
        # mpmodel.run_evaluation()
        print("run evaluation")

    # Save data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mapping Prejudice Project, Racial Covenant Identification Pipeline')

    parser.add_argument('--file_input', type=str,
                        help="Location of input file")
    parser.add_argument('--identification_method', choices=['fuzzy', 'er'],
                        help="Location of input file")

    parser.add_argument('--file_output', type=str,
                       help="Output filename")
    
    parser.add_argument('--dev',
                        action='store_true')
    
    
    args = parser.parse_args()

    main()