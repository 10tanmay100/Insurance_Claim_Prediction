from src.insurance_fraud.exception import InsuranceFraudException
import os,sys

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