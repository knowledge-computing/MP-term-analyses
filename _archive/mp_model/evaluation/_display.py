from typing import List
from tabulate import tabulate
import matplotlib.pyplot

def result_table(data:List[list], headers:List[str], 
                 table_format:str="psql") -> int:
    
    try: 
        print(tabulate(table=data, headers=headers,
                       tablefmt=table_format))
        return 0
    except:
        print("Tabulate failed")
        print(data)
        return -1

def result_bar():
    return 0