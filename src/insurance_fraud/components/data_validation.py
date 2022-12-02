from src.insurance_fraud.constant.training_pipeline import SCHEMA_FILE_PATH
from src.insurance_fraud.entity import DataIngestionArtifact, DataValidationArtifact,DataValidationConfig
from src.insurance_fraud.utils.main_utils import read_yaml_file,write_yaml_file
from src.insurance_fraud.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
import os,sys
import pandas as pd



class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.__schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceFraudException(e,sys) from e

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            logging.info("Number of columns validation started!!!")
            number_of_columns=len(self.__schema_config["columns"])
            if len(dataframe.columns)==number_of_columns:
                    logging.info(f"data frame columns and number of columns checking passed...->> {len(dataframe.columns)==number_of_columns}")
                    logging.info("Number of columns validation ended!!!")
                    return True
            else:
                    logging.info(f"data frame columns and number of columns checking failed...->> {len(dataframe.columns)==number_of_columns}")
                    logging.info("Number of columns validation ended!!!")
                    return False
        except Exception as e:
            logging.error("validate number of columns check has some issue..")
            raise InsuranceFraudException(e,sys) from e


    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            logging.info("Existence of numerical column checking has been started!!")
            numerical_columns=self.__schema_config["numerical_columns"]
            logging.info(f"Number of columns {numerical_columns}")
            dataframe_columns=dataframe.columns
            missing_numerical_columns_status=False
            missing_column_names=[]
            for cols in numerical_columns:
                if cols not in dataframe_columns:
                    missing_numerical_columns_status=True
                    missing_column_names.append(cols)
            logging.info(f"Missing numerical columns are [{missing_column_names}]")
            return missing_numerical_columns_status
        except Exception as e:
            raise InsuranceFraudException(e,sys) from e

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise InsuranceFraudException(e,sys) from e
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.5):
        try:
            logging.info("Data drift checking started..")
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                logging.info("Data Distribution checking has been started!!")
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                    logging.info(f"Data Distribution checking has been started!! {is_found}")
                else:
                    is_found=True
                    logging.info(f"Data Distribution checking has been started!! {is_found}")
                    status=False
                report.update({column:{"p_value":float(is_same_dist.pvalue),"drift_status":is_found}})
            

            #defining the drift report file path
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            logging.info(f"Drift report file path has been defined path is -> {drift_report_file_path}")

            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(drift_report_file_path,report,replace=True)
            return status


        except Exception as e:
            raise InsuranceFraudException(e,sys) from e


    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info("Data validation stage initialization started!!")
            error_msg=""
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            #reading train and test file location
            train_dataframe=DataValidation.read_data(train_file_path)
            logging.info(f"Reading the train file path is -> {train_dataframe}")
            test_dataframe=DataValidation.read_data(test_file_path)
            logging.info(f"Reading the test file path is -> {test_dataframe}")



            #validation number of columns
            train_status=self.validate_number_of_columns(train_dataframe)
            if not train_status:
                error_msg=f"{error_msg}Train dataframe does not contain all columns"
            test_status=self.validate_number_of_columns(test_dataframe)
            if not test_status:
                error_msg=f"{error_msg}Test dataframe does not contain all columns"
            

            #validate numerical col exist
            train_status=self.is_numerical_column_exist(dataframe=train_dataframe)
            if train_status:
                error_msg=f"{error_msg}Train dataframe does not contain all numeric columns"
            test_status=self.is_numerical_column_exist(dataframe=test_dataframe)
            if test_status:
                error_msg=f"{error_msg}Test dataframe does not contain all numeric columns"


            if len(error_msg) > 0:
                    try:      
                              logging.info(f"error msg detected -->{error_msg}")
                              os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path),exist_ok=True)
                    
                              logging.info(f"{os.path.dirname(self.data_validation_config.invalid_train_file_path)} directory created")

                              train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path)

                              os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path),exist_ok=True)

                              logging.info(f"{os.path.dirname(self.data_validation_config.invalid_test_file_path)} directory created")

                              test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path)
                    except:
                              raise Exception(error_msg)

            #check data drift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            if not os.path.exists(self.data_validation_config.invalid_train_file_path):
                train_path_invalid=None
            else:
                train_path_invalid=self.data_validation_config.invalid_train_file_path
            
            if not os.path.exists(self.data_validation_config.invalid_test_file_path):
                test_path_invalid=None
            else:
                test_path_invalid=self.data_validation_config.invalid_test_file_path




            data_validation_artifact=DataValidationArtifact(
                    validation_status=status,valid_train_file_path=self.data_ingestion_artifact.train_file_path,valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                    invalid_train_file_path=train_path_invalid,invalid_test_file_path=test_path_invalid,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact


        except Exception as e:
            raise InsuranceFraudException(e,sys) from e
