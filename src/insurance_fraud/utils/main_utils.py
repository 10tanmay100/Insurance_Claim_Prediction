from src.insurance_fraud.exception import InsuranceFraudException
import yaml,sys,os
import pandas as pd
import numpy as np
from pandas import DataFrame
import dill
from scipy import stats

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
        


def save_numpy_array(file_path:str,array:np.array)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e


def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e


def save_object(file_path:str,obj:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as pickle_obj:
            dill.dump(obj,pickle_obj)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e



def load_object(file_path:str)->None:
    try:
        if os.path.exists(file_path):
            with open(file_path,"rb") as pickle_obj:
                return dill.load(pickle_obj)
        else:
            raise Exception("file path does not exist: %s" % file_path)
    except Exception as e:
        raise InsuranceFraudException(e,sys) from e


def handiling_outlier(dataframe:DataFrame,col_name)-> DataFrame:
    z = pd.DataFrame(np.abs(stats.zscore(dataframe[col_name])))
    z.columns=["z"]
    df=pd.concat([dataframe,z],axis=1)
    df=df[df["z"]<3]
    df=df.drop(["z"],axis=1)
    return df

def extracting_feature(dataframe:DataFrame, old_col_name:str,new_col_name:str,unit:str,pos:int)->DataFrame:
    """
    This function will extract features from the dataframe
    daraframe will take a pandas dataframe as an input and old col name will be that column name from which we want to extract the features
    New col name will be the new name of that extracted feature
    Unit will be the unit on which we want to split
    """
    dataframe[new_col_name]=0
    for idx in list(dataframe[old_col_name].str.split("@").index):
        dataframe.loc[idx,new_col_name]=float(dataframe.loc[idx,old_col_name].split("@")[pos].split(unit)[0])
    return dataframe


def handiling_categorical_features(dataframe:DataFrame,col_name:str):
    #handiling segment categorical column
    features=list(dataframe[col_name].value_counts(ascending=False)[:3].index)
    for rows in range(dataframe.shape[0]):
        if dataframe.loc[rows,col_name] not in features:
            dataframe.loc[rows,col_name]="Others"
    return dataframe

