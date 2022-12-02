from src.insurance_fraud.exception import InsuranceFraudException
import yaml,sys,os
import pandas as pd
def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e


def write_yaml_file(file_path:str,content:object,replace:bool)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as f:
            yaml.dump(content,f)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e