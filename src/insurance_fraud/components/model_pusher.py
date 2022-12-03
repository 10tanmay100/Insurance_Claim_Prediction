from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity.artifact_entity import *
from src.insurance_fraud.entity.config_entity import *
import os,sys
import shutil
from src.insurance_fraud.ml.metrics.classification_metrics import get_classification_metrics
from src.insurance_fraud.ml.models.model_estimator import InsuranceModel
from src.insurance_fraud.utils.main_utils import load_object,save_object,write_yaml_file


class ModelPusher:

          def __init__(self,model_pusher_config:ModelPusherConfig,model_eval_artifact:ModelEvaluationArtifact):
                    try:
                              self.model_pusher_config=model_pusher_config
                              self.model_eval_artifact=model_eval_artifact
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          
          def initiate_model_pusher(self)->ModelPusherArtifact:
                    try:
                              trained_model_path=self.model_eval_artifact.trained_model_file_path
                              #creating model directory
                              model_file_path=self.model_pusher_config.model_pusher_model_file_path
                              os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
                              shutil.copy(src=trained_model_path,dst=model_file_path)
                              #store in saved model direcotory too
                              saved_model_path=self.model_pusher_config.saved_model_path
                              os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
                              shutil.copy(src=trained_model_path,dst=saved_model_path)

                              model_pusher_artifact=ModelPusherArtifact(saved_model_path=saved_model_path,model_file_path=model_file_path,preprocessor_file_path=self.model_eval_artifact.preprocessor_artifact)
                              return model_pusher_artifact
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e



