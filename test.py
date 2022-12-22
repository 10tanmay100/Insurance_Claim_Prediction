from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity import *
from src.insurance_fraud.pipeline.training_pipeline import TrainPipeline
from src.insurance_fraud.ml.models.model_estimator import ModelResolver
from src.insurance_fraud.utils.main_utils import load_object
from src.insurance_fraud.constant.training_pipeline import *



training_pipeline=TrainPipeline()
model_pusher_artifact=training_pipeline.run_pipeline()
print(model_pusher_artifact)