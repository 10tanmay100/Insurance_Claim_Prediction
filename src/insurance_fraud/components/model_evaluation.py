from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity.artifact_entity import *
from src.insurance_fraud.entity.config_entity import *
import os,sys
from src.insurance_fraud.ml.metrics.classification_metrics import get_classification_metrics
from src.insurance_fraud.ml.models.model_estimator import InsuranceModel
from src.insurance_fraud.utils.main_utils import load_object,save_object,write_yaml_file,load_numpy_array_data
from src.insurance_fraud.ml.models.model_estimator import ModelResolver
import pandas as pd
import numpy as np

class ModelEvaluation:

          def __init__(self,model_eval_config:ModelEvaluationConfig,data_transformation_artifact:DataTransformationArtifact,model_trainer_artifact:ModelTrainerArtifact):
                    try:
                              self.model_eval_config=model_eval_config
                              self.data_transformation_artifact=data_transformation_artifact
                              self.model_trainer_artifact=model_trainer_artifact
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          
          def initiate_model_evaluation(self)->ModelEvaluationArtifact:
                    try:      #takig the valid train and test file
                              logging.info("Model Evaluation stage initialized......")
                              transformed_train_file_path=self.data_transformation_artifact.transformed_train_file_path
                              transformed_test_file_path=self.data_transformation_artifact.transformed_test_file_path
                              #taking training and testing df
                              train_df=pd.read_csv(transformed_train_file_path)
                              logging.info("Reading train file path")
                              test_df=pd.read_csv(transformed_test_file_path)
                              logging.info("Reading test file path")


                              df=pd.concat([train_df,test_df],axis=0)
          
                              logging.info("Concatenating training and testing")
                              #taking the model
                              train_model_file_path=self.model_trainer_artifact.trained_model_file_path
                              # model=load_object(file_path=train_model_file_path)
                              logging.info("Load model object ....")

                              model_resolver=ModelResolver()
                              if not model_resolver.is_model_exist():
                                        logging.info("Since model does not exist...we return the artifact")
                                        model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=True,changed_accuracy=None,
                                        best_model_path=None,
                                        trained_model_file_path=train_model_file_path,
                                        trained_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,best_model_metric_artifact=None,preprocessor_artifact=self.data_transformation_artifact.transformed_object_file_path)
                                        return model_evaluation_artifact
                              latest_model_path=model_resolver.get_best_model_path()
                              latest_model=load_object(file_path=latest_model_path)
                              train_model=load_object(file_path=train_model_file_path)

                              y_true=df[TARGET_COLUMN]
                              df=df.drop(columns=[TARGET_COLUMN],axis=1)
                              y_train_pred=train_model.predict(df)
                              y_latest_pred=latest_model.predict(df)


                              trained_metric=get_classification_metrics(y_true,y_train_pred)
                              latest_metric=get_classification_metrics(y_true,y_latest_pred)

                              improved_accuracy=trained_metric.f1_Score-latest_metric.f1_Score
                              if improved_accuracy>=self.model_eval_config.model_evaluator_threshold:
                                         model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=True,changed_accuracy=improved_accuracy,
                                        best_model_path=latest_model_path,
                                        trained_model_file_path=train_model_file_path,
                                        trained_model_metric_artifact=trained_metric,best_model_metric_artifact=latest_metric,preprocessor_artifact=self.data_transformation_artifact.transformed_object_file_path)
                                        
                              else:
                                        model_evaluation_artifact=ModelEvaluationArtifact(is_model_accepted=False,changed_accuracy=improved_accuracy,
                                        best_model_path=latest_model_path,
                                        trained_model_file_path=train_model_file_path,
                                        trained_model_metric_artifact=trained_metric,best_model_metric_artifact=latest_metric,preprocessor_artifact=self.data_transformation_artifact.transformed_object_file_path)
                              model_eval_report=model_evaluation_artifact.__dict__
                              write_yaml_file(file_path=self.model_eval_config.report_file_path,content=model_eval_report,replace=True)
                              logging.info("Evaluation Report has been written....")
                              return model_evaluation_artifact

                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          

          

