from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity import *
from src.insurance_fraud.pipeline.training_pipeline import TrainPipeline




if __name__ == '__main__':
    training_pipeline=TrainPipeline()
    print(training_pipeline.run_pipeline())