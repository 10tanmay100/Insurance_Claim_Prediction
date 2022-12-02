from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity import *
import pandas as pd
from pandas import DataFrame
from src.insurance_fraud.data_access.insurance_data import InsuranceFraudData
import os,sys
from sklearn.model_selection import train_test_split


class DataIngestion:
          def __init__(self,data_ingestion_config:DataIngestionConfig):
                    self.data_ingestion_config=data_ingestion_config
          def export_data_into_feature_store(self)->DataFrame:
                    """
                    Export data into feature store from cassandra
                    """
                    try:
                              logging.info("export data into feature store")
                              insurance_data=InsuranceFraudData()
                              dataframe=insurance_data.export_data_as_dataframe()

                              feature_store_file_path=self.data_ingestion_config.feature_store_file_path

                              #creating feature store folders
                              dir_path=os.path.dirname(feature_store_file_path)
                              logging.info(f"creating feature store path and dir name {dir_path}")
                              os.makedirs(dir_path,exist_ok=True)
                              dataframe.to_csv(feature_store_file_path,index=False)
                              logging.info(f"Dataframe sent as csv in the path ->{feature_store_file_path}")
                              return dataframe
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          def split_data_as_train_test(self,dataframe:DataFrame)->None:
                    """
                    Split the data from feature store
                    """
                    try:
                              train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=self.data_ingestion_config.random_state)
                              logging.info("Dividing the dataset as train and test")
                              dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
                              os.makedirs(dir_path,exist_ok=True)
                              logging.info(f"Directory created ->{dir_path}")
                              train_set.to_csv(self.data_ingestion_config.training_file_path,index=False)
                              logging.info(f"Sending train csv to path -> {self.data_ingestion_config.training_file_path}")
                              test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False)
                              logging.info(f"Sending test csv to path -> {self.data_ingestion_config.testing_file_path}")
                    except Exception as e:
                              logging.error(f"Error in data ingestion component and error is {e}")
                              raise InsuranceFraudException(e,sys) from e
          
          def initiate_data_ingestion(self)->DataIngestionArtifact:
                    try:
                              logging.info("Data Ingestion stage initialization started!!")
                              dataframe=self.export_data_into_feature_store()
                              self.split_data_as_train_test(dataframe)
                              data_ingestion_artifact=DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)
                              logging.info("Data Ingestion stage initialization ended!!")
                              return data_ingestion_artifact
                    except Exception as e:
                              logging.error(f"Error in data ingestion stage initializatio {e}")
                              raise InsuranceFraudException(e,sys) from e



