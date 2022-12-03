from datetime import datetime
from src.insurance_fraud.constant import *
from src.insurance_fraud.logger import logging
from pathlib import Path
import os


class TrainingPipelineConfig:
          def __init__(self,timestamp=datetime.now()):
                    timestamp=datetime.now().strftime("%m_%d_%Y_%M_%S")
                    self.pipeline_name:str=PIPELINE_NAME
                    self.artifact_dir:str=os.path.join(ARTIFACT_DIR,timestamp)
                    self.timestamp:str=timestamp

class DataIngestionConfig:
          def __init__(self,training_pipeline_config:TrainingPipelineConfig):
                    #data ingestion directory path
                    logging.info("Data Ingestion Config has been started!!")

                    self.data_ingestion_dir:Path=os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)

                    logging.info(f"Data Ingestion directory path defined!! path is {self.data_ingestion_dir}")
                    #feature store file path defined
                    self.feature_store_file_path:Path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)

                    logging.info(f"Feature Store file path defined!! path is {self.feature_store_file_path}")
                    #training file path defined
                    self.training_file_path:Path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)

                    logging.info(f"Training file path defined!! path is {self.training_file_path}")

                    #testing file path defined
                    self.testing_file_path:Path=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)

                    logging.info(f"Testing file path defined!! path is {self.testing_file_path}")
                    #train test split ratio
                    self.train_test_split_ratio:float=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

                    logging.info(f"Defined train test split.... {self.train_test_split_ratio}")
                    #random state defined
                    self.random_state:int=DATA_INGESTION_RANDOM_STATE
                    
                    logging.info(f"Defined random state {self.random_state}")

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        #data validation directory path
        logging.info("Data validation Config has been started!!")

        self.data_validation_dir:Path=os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)

        logging.info(f"Data validation directory path defined!! path is {self.data_validation_dir}")
        #defining valid data directory
        self.valid_data_directory:Path=os.path.join(self.data_validation_dir,DATA_VALIDATION_VALID_DIR)

        logging.info(f"Data valid data directory path defined!! path is {self.valid_data_directory}")
        #defining invalid data directory
        self.invalid_data_directory:Path=os.path.join(self.data_validation_dir,DATA_VALIDATION_INVALID_DIR)

        logging.info(f"Data invalid data directory path defined!! path is {self.invalid_data_directory}")
        #defining the valid train file path
        self.valid_train_file_path:Path=os.path.join(self.valid_data_directory,TRAIN_FILE_NAME)

        logging.info(f"Data valid data train path defined!! path is {self.valid_train_file_path}")
        #defining the invalid train file path
        self.invalid_train_file_path:Path=os.path.join(self.invalid_data_directory,TRAIN_FILE_NAME)

        logging.info(f"Data invalid data train path defined!! path is {self.invalid_train_file_path}")
        #defining the valid test file path
        self.valid_test_file_path:Path=os.path.join(self.valid_data_directory,TEST_FILE_NAME)

        logging.info(f"Data valid data test path defined!! path is {self.valid_test_file_path}")
        #defining the invalid test file path
        self.invalid_test_file_path:Path=os.path.join(self.invalid_data_directory,TEST_FILE_NAME)

        logging.info(f"Data invalid data test path defined!! path is {self.invalid_test_file_path}")
        #defining drift report file path
        self.drift_report_file_path:Path=os.path.join(self.data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

        logging.info(f"Data drift report file path defined!! path is {self.drift_report_file_path}")



class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        #defining the root directory of data transformation
        logging.info("Data Transformation Config has been started!!")

        self.data_transformation_dir:Path=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)

        logging.info(f"Data transformation directory path defined!! path is {self.data_transformation_dir}")
        #defining the train file path
        self.data_transformed_train_dir:Path=os.path.join(self.data_transformation_dir,TRAIN_FILE_NAME)

        logging.info(f"Train file path defined in data transformation stage and csv file extension and file and path is -> {self.data_transformed_train_dir}")
        #defining the test file path
        self.data_transformed_test_dir:Path=os.path.join(self.data_transformation_dir,TEST_FILE_NAME)

        logging.info(f"Test file path defined in data transformation stage and csv file extension and file and path is -> {self.data_transformed_test_dir}")
        #defining the directory to store the transformation pickle file in my system
        self.data_transformation_transformed_obj_dir:Path=os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)

        logging.info(f"Preprocessor pickle file's path defined and path is -> {self.data_transformation_transformed_obj_dir}")



class ModelBuilderConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        logging.info("Model Builder Config has been started!!")
        #defining the root diretory of model building
        self.model_trainer_dir:Path=os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)

        logging.info(f"Trained Directory created , {self.model_trainer_dir}")
        #defining the model store path
        self.model_trained_dir:Path=os.path.join(self.model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAIN_MODEL_NAME)

        logging.info(f"Trained model stored Directory created , {self.model_trained_dir}")
        #defining the model accuracy threshold
        self.model_threshold_accuracy:float=MODEL_TRAINER_EXPECTED_SCORE

        logging.info(f"Threshold accuracy defined {self.model_threshold_accuracy}")
        #defining overfitting threshold
        self.model_over_underfit_thershold_diff:float=OVERFITTING_UNDERFITTING_THRESHOLD
        logging.info(f"Overfitting threshold defined {self.model_over_underfit_thershold_diff}")


class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        logging.info("Model Evaluator Config has been started!!")
        #Defining the root directory for model evaluation
        self.model_evaluator_dir=os.path.join(training_pipeline_config.artifact_dir,MODEL_EVALUATION_DIR_NAME)

        logging.info(f"Model evaluation directory defined {self.model_evaluator_dir}")
        #defining the report file path
        self.report_file_path=os.path.join(self.model_evaluator_dir,MODEL_EVALUATION_REPORT_FILE)

        logging.info(f"Evaluate model file path defined {self.report_file_path}")
        #defining the evalutor thereshold if new model crosses that we choose the new one
        self.model_evaluator_threshold=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

        logging.info("Model Evaluator threshold defined..")

class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        logging.info("Model Pusher config has been started")
        #defining the model pusher root directory
        self.model_pusher_dir=os.path.join(training_pipeline_config.artifact_dir,MODEL_PUSHER_DIR_NAME)

        logging.info("Model Pusher directory created..")
        #defining the model pusher folder model file path
        self.model_pusher_model_file_path=os.path.join(self.model_pusher_dir,MODEL_NAME)

        logging.info("Model Pushiing model file created...")
        #defining the model pusher in saved model
        timestamp=round(datetime.now().timestamp())
        self.saved_model_path=os.path.join(SAVED_MODEL_DIR,str(timestamp),MODEL_NAME)
        logging.info(f"New Model saved in  {self.saved_model_path}")