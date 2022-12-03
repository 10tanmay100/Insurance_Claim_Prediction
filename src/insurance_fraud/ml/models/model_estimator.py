from src.insurance_fraud.exception import InsuranceFraudException
import os,sys
from src.insurance_fraud.constant.training_pipeline import *

class InsuranceModel:
          def __init__(self,preprocessor,model):
                    self.preprocessor=preprocessor
                    self.model=model
          
          def predict(self,X):
                    try:
                              x_transform=self.preprocessor.transform(X)
                              y_hat=self.model.predict(x_transform)
                              return y_hat
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e


class ModelResolver:
          def __init__(self,model_dir=SAVED_MODEL_DIR):

                    try:
                              self.model_dir=model_dir
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e

          def get_best_model_path(self):
                    try:
                              timestamps=list(map(int,os.listdir(self.model_dir)))
                              latest_timestamp=max(timestamps)
                              latest_model_path=os.path.join(self.model_dir,str(latest_timestamp),MODEL_FILE_NAME)
                              return latest_model_path
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e

          def is_model_exist(self)->str:
                    try:
                              if not os.path.exists(self.model_dir):
                                        return False
                              timestamps=os.listdir(self.model_dir)
                              if len(timestamps) ==0:
                                        return False
                              latest_model_path=self.get_best_model_path()
                              if not os.path.exists(latest_model_path):
                                        return False
                              return True
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e