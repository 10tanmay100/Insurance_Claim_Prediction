import os
#defining common constant variables for trainig pipeline
TARGET_COLUMN="is_claim"
PIPELINE_NAME:str="insurance_fraud"
ARTIFACT_DIR:str="artifact"
FILE_NAME:str="insurance.csv"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"
MODEL_FILE_NAME:str="model.pkl"
SCHEMA_FILE_PATH:str=os.path.join("config","schema.yaml")