from src.insurance_fraud.utils.main_utils import load_numpy_array_data
from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity.artifact_entity import *
from src.insurance_fraud.entity.config_entity import *
from xgboost import XGBClassifier
import os,sys
from src.insurance_fraud.constant.training_pipeline import TARGET_COLUMN
import pandas as pd
from src.insurance_fraud.ml.metrics.classification_metrics import get_classification_metrics
from src.insurance_fraud.ml.models.model_estimator import InsuranceModel
from src.insurance_fraud.utils.main_utils import load_object,save_object



class ModelTrainer:
          def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelBuilderConfig):
                    try:
                              self.data_transformation_artifact = data_transformation_artifact
                              self.model_trainer_config=model_trainer_config

                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          def train_model(self,X_train,y_train):
                    try:
                              xgb_clf=XGBClassifier()
                              xgb_clf.fit(X_train,y_train)
                              return xgb_clf
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e

          def initiate_model_trainer(self)->ModelTrainerArtifact:
                    try:      #loading training array and testing array
                              logging.info("Model Building Stage Initialized")
                              # train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
                              train_arr=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
                              logging.info("Loading train data..")
                              # test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
                              test_arr=pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
                              logging.info("Loading test data..")
                              #spltting the data
                              X_train=train_arr.drop([TARGET_COLUMN],axis=1)
                              y_train=train_arr[TARGET_COLUMN]
                              X_test=test_arr.drop([TARGET_COLUMN],axis=1)
                              y_test=test_arr[TARGET_COLUMN]
                              logging.info("Splitting the data as train and test")
                              #model training
                              model=self.train_model(X_train,y_train)
                              logging.info("Model training has been started!!!")
                              y_train_pred=model.predict(X_train)
                              classification_metrics_train=get_classification_metrics(y_train,y_train_pred)
                              logging.info(f"Training metrics achieved!,{classification_metrics_train}")
                              y_test_pred=model.predict(X_test)
                              classification_metrics_test=get_classification_metrics(y_test,y_test_pred)
                              logging.info(f"Testing metrics achieved! {classification_metrics_test}")

                              if classification_metrics_train.f1_Score < self.model_trainer_config.model_threshold_accuracy:
                                        logging.error("f1 score is not acceptable!!")
                                        raise Exception("f1 score is not acceptable!!")

                              #overfitting and underfitting
                              diff=abs(classification_metrics_train.f1_Score-classification_metrics_train.f1_Score)
                              logging.info(f"F1 score difference betwen train and test {diff}")

                              if diff>self.model_trainer_config.model_over_underfit_thershold_diff:
                                        raise Exception("Model Might be overfitted or underfitted")

                              preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)
                              logging.info("Loading the preprocessor object...")

                              model_dir_path=os.path.dirname(self.model_trainer_config.model_trained_dir)
                              os.makedirs(model_dir_path,exist_ok=True)
                              logging.info("Train model directory created!!!!")

                              insurance_model=InsuranceModel(preprocessor=preprocessor,model=model)
                              save_object(file_path=self.model_trainer_config.model_trained_dir,obj=insurance_model)
                              logging.info("Model object has been created..")


                              return ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.model_trained_dir,trained_metric_artifact=classification_metrics_train,test_metric_artifact=classification_metrics_test)

                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e


