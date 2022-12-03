from src.insurance_fraud.entity.artifact_entity import ClassificationMetricsArtifact
from src.insurance_fraud.exception import InsuranceFraudException
import os,sys
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score

def get_classification_metrics(y_true,y_pred)->ClassificationMetricsArtifact:
          try:
                    model_f1=f1_score(y_true,y_pred)
                    model_precision_score=precision_score(y_true,y_pred)
                    model_recall_score=recall_score(y_true,y_pred)
                    model_auc=roc_auc_score(y_true,y_pred)
                    return ClassificationMetricsArtifact(f1_Score=model_f1,precision_Score=model_precision_score,recall_Score=model_recall_score,roc_auc_Score=model_auc)
          except Exception as e:
                    raise InsuranceFraudException(e,sys) from e