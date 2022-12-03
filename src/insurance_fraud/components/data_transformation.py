import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.insurance_fraud.constant.training_pipeline import TARGET_COLUMN
from src.insurance_fraud.constant.training_pipeline.data_transformation_constants import AFTER_EXTRACTION_UNNECESSARY_COLS
from src.insurance_fraud.entity.artifact_entity import (DataTransformationArtifact,DataValidationArtifact,)
from src.insurance_fraud.entity.config_entity import DataTransformationConfig
from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.logger import logging
from src.insurance_fraud.utils.main_utils import save_numpy_array, save_object,handiling_outlier,extracting_feature,handiling_categorical_features



class DataTransformation:
          def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
                    try:
                              self.data_validation_artifact=data_validation_artifact
                              self.data_tranformation_config=data_transformation_config
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e

          @staticmethod
          def read_data(file_path):
                    try:
                              return pd.read_csv(file_path)
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          @classmethod
          def get_data_transformer_object(cls)->Pipeline:
                    try:
                              standard_scaler=StandardScaler()
                              preprocessor=Pipeline(
                                        steps=[('StandardScaler',standard_scaler)]
                              )
                              return preprocessor
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          
          def initiate_data_transformation(self)->DataTransformationArtifact:
                    try:

                              logging.info("Data transformation stage initialization stage has been started!!!")
                              train_dataframe=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
                              logging.info(f"Reading training dataframe path is -> {self.data_validation_artifact.valid_train_file_path}")
                              test_dataframe=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
                              preprocessor=self.get_data_transformer_object()
                              logging.info(f"Reading testing dataframe path is -> {self.data_validation_artifact.valid_train_file_path}")
                              #training dataframe
                              logging.info("Applying some preprocessing in the training data")
                              logging.info("Handiling the outliers.. training..")
                              train_dataframe=handiling_outlier(train_dataframe,"age_of_car")
                              train_dataframe=handiling_outlier(train_dataframe,"age_of_policyholder")
                              logging.info("After Handiling the outlier in the data let's rese the index..")
                              train_dataframe=train_dataframe.reset_index(drop=True)

                              logging.info("Feature Extraction started!!!!")
                              logging.info("Column name--> {max_torque}")
                              train_dataframe=extracting_feature(dataframe=train_dataframe,old_col_name="max_torque",new_col_name="torque_nm",unit="Nm",pos=0)
                              train_dataframe=extracting_feature(dataframe=train_dataframe,old_col_name="max_torque",new_col_name="torque_rpm",unit="rpm",pos=1)


                              logging.info("Column name--> {max_power}")
                              train_dataframe=extracting_feature(dataframe=train_dataframe,old_col_name="max_power",new_col_name="power_bhp",unit="bhp",pos=0)
                              train_dataframe=extracting_feature(dataframe=train_dataframe,old_col_name="max_power",new_col_name="power_rpm",unit="rpm",pos=1)


                              #dropping unnecessary categorical columns
                              train_dataframe=train_dataframe.drop(AFTER_EXTRACTION_UNNECESSARY_COLS,axis=1)
                              logging.info(f"dropping unnecessary columns -> {AFTER_EXTRACTION_UNNECESSARY_COLS}")

                              #handiling categorical columns
                              train_dataframe=handiling_categorical_features(train_dataframe,col_name="segment")
                              logging.info("Handiling the categorical column which is segment")
                              train_dataframe=handiling_categorical_features(train_dataframe,col_name="model")
                              logging.info("Handiling the categorical column which is model")


                              #converting remaining categoricals to numerical
                              train_dataframe["is_esc"]=train_dataframe["is_esc"].map({'Yes':1,'No':0})
                              train_dataframe["is_adjustable_steering"]=train_dataframe["is_adjustable_steering"].map({'Yes':1,'No':0})
                              train_dataframe["is_parking_sensors"]=train_dataframe["is_parking_sensors"].map({'Yes':1,'No':0})
                              train_dataframe["is_parking_camera"]=train_dataframe["is_parking_camera"].map({'Yes':1,'No':0})
                              train_dataframe["transmission_type"]=train_dataframe["transmission_type"].map({'Manual':1,'Automatic':0})
                              train_dataframe["is_brake_assist"]=train_dataframe["is_brake_assist"].map({'Yes':1,'No':0})
                              train_dataframe["is_central_locking"]=train_dataframe["is_central_locking"].map({'Yes':1,'No':0})
                              train_dataframe["is_power_steering"]=train_dataframe["is_power_steering"].map({'Yes':1,'No':0})
                              logging.info("Handled all categorical features...")


                              #applying get dummies to do OHE
                              train_dataframe=pd.concat([train_dataframe,pd.get_dummies(train_dataframe[["segment","model","fuel_type"]],drop_first=True)],axis=1)
                              logging.info("Applying OHE to train_dataframe")
                              #dropping those normal non OHE cols
                              train_dataframe=train_dataframe.drop(["segment","model","fuel_type"],axis=1)
                              logging.info("Dropping ['segment','model','fuel_type']")

                              logging.info("Now splitting the input and target feature for trainining the model")
                              input_feature_train_df=train_dataframe.drop(columns=[TARGET_COLUMN])
                              logging.info(f"Glimpse of input feature data {input_feature_train_df.head()}")
                              target_feature_train_df=train_dataframe[TARGET_COLUMN]
                              logging.info(f"Glimpse of target feature data {target_feature_train_df.head()}")

                              print(input_feature_train_df)
                              

                              #getting the preprocessor obj
                              preprocessor_obj=preprocessor.fit(input_feature_train_df)
                              logging.info("Fitting preprocessor on the input feature data!!")
                              save_object(file_path=self.data_tranformation_config.data_transformation_transformed_obj_dir,obj=preprocessor_obj)
                              logging.info(f"Saving the preprocessor object in the path ->> {self.data_tranformation_config.data_transformation_transformed_obj_dir}")
                              transformed_feature_train_df=pd.DataFrame(preprocessor.transform(input_feature_train_df),columns=input_feature_train_df.columns)

                              print(transformed_feature_train_df)
                              logging.info("Transforming the input train feature......")
                              #testing dataframe
                              logging.info("Applying some preprocessing in the testing data")
                              logging.info("Handiling the outliers.. testing....")
                              test_dataframe=handiling_outlier(test_dataframe,"age_of_car")
                              test_dataframe=handiling_outlier(test_dataframe,"age_of_policyholder")
                              logging.info("After Handiling the outlier in the data let's reset the index..")
                              test_dataframe=test_dataframe.reset_index(drop=True)

                              logging.info("Feature Extraction started!!!!")
                              logging.info("Column name--> {max_torque}")
                              test_dataframe=extracting_feature(dataframe=test_dataframe,old_col_name="max_torque",new_col_name="torque_nm",unit="Nm",pos=0)
                              test_dataframe=extracting_feature(dataframe=test_dataframe,old_col_name="max_torque",new_col_name="torque_rpm",unit="rpm",pos=1)


                              logging.info("Column name--> {max_power}")
                              test_dataframe=extracting_feature(dataframe=test_dataframe,old_col_name="max_power",new_col_name="power_bhp",unit="bhp",pos=0)
                              test_dataframe=extracting_feature(dataframe=test_dataframe,old_col_name="max_power",new_col_name="power_rpm",unit="rpm",pos=1)


                              #dropping unnecessary categorical columns
                              test_dataframe=test_dataframe.drop(AFTER_EXTRACTION_UNNECESSARY_COLS,axis=1)
                              logging.info(f"dropping unnecessary columns -> {AFTER_EXTRACTION_UNNECESSARY_COLS}")

                              #handiling categorical columns
                              test_dataframe=handiling_categorical_features(test_dataframe,col_name="segment")
                              logging.info("Handiling the categorical column which is segment")
                              test_dataframe=handiling_categorical_features(test_dataframe,col_name="model")
                              logging.info("Handiling the categorical column which is model")


                              #converting remaining categoricals to numerical
                              test_dataframe["is_esc"]=test_dataframe["is_esc"].map({'Yes':1,'No':0})
                              test_dataframe["is_adjustable_steering"]=test_dataframe["is_adjustable_steering"].map({'Yes':1,'No':0})
                              test_dataframe["is_parking_sensors"]=test_dataframe["is_parking_sensors"].map({'Yes':1,'No':0})
                              test_dataframe["is_parking_camera"]=test_dataframe["is_parking_camera"].map({'Yes':1,'No':0})
                              test_dataframe["transmission_type"]=test_dataframe["transmission_type"].map({'Manual':1,'Automatic':0})
                              test_dataframe["is_brake_assist"]=test_dataframe["is_brake_assist"].map({'Yes':1,'No':0})
                              test_dataframe["is_central_locking"]=test_dataframe["is_central_locking"].map({'Yes':1,'No':0})
                              test_dataframe["is_power_steering"]=test_dataframe["is_power_steering"].map({'Yes':1,'No':0})


                              #applying get dummies to do OHE
                              test_dataframe=pd.concat([test_dataframe,pd.get_dummies(test_dataframe[["segment","model","fuel_type"]],drop_first=True)],axis=1)
                              logging.info("Applying OHE to test_dataframe")
                              #dropping those normal non OHE cols
                              test_dataframe=test_dataframe.drop(["segment","model","fuel_type"],axis=1)
                              logging.info("Dropping ['segment','model','fuel_type']")


                              logging.info("Now splitting the input and target feature for testing the model")
                              input_feature_test_df=test_dataframe.drop(columns=[TARGET_COLUMN])
                              logging.info(f"Glimpse of input feature data {input_feature_test_df.head()}")
                              target_feature_test_df=test_dataframe[TARGET_COLUMN]
                              logging.info(f"Glimpse of target feature data {target_feature_test_df.head()}")


                              transformed_feature_test_df=pd.DataFrame(preprocessor.transform(input_feature_test_df),columns=input_feature_test_df.columns)
                              print(transformed_feature_test_df)
                              logging.info("Transforming the input test feature......")


                              smt=SMOTETomek()
                              logging.info("Imbalance handiling object defined")
                              input_feature_train_final,target_feature_train_final=smt.fit_resample(transformed_feature_train_df,target_feature_train_df)
                              logging.info("Fit resample applied on training data!!")
                              
                              
                              input_feature_test_final,target_feature_test_final=smt.fit_resample(transformed_feature_test_df,target_feature_test_df)
                              logging.info("Fit resample applied on testing data!!")

                              # transformed_train_arr=np.c_[input_feature_train_final,np.array(target_feature_train_final)]
                              transformed_train_df=pd.concat([pd.DataFrame(input_feature_train_final),target_feature_train_final],axis=1)
                              print(transformed_train_df)
                              logging.info("Concatenating transformed train data...")
                              
                              transformed_test_df=pd.concat([pd.DataFrame(input_feature_test_final),target_feature_test_final],axis=1)
                              logging.info("Concatenating transformed test data...")
                              print(transformed_test_df)

                              #saving numpy arrays
                              # save_numpy_array(file_path=self.data_tranformation_config.data_transformed_train_dir,array=transformed_train_arr)
                              transformed_train_df.to_csv(self.data_tranformation_config.data_transformed_train_dir,index=False)
                              logging.info(f"Saving transformed train array in the specified path -> {self.data_tranformation_config.data_transformed_train_dir}")
                              # save_numpy_array(file_path=self.data_tranformation_config.data_transformed_test_dir,array=transformed_test_arr)
                              transformed_test_df.to_csv(self.data_tranformation_config.data_transformed_test_dir,index=False)
                              logging.info(f"Saving transformed test array in the specified path -> {self.data_tranformation_config.data_transformed_test_dir}")

                              data_transformation_artifact=DataTransformationArtifact(transformed_object_file_path=self.data_tranformation_config.data_transformation_transformed_obj_dir,transformed_train_file_path=self.data_tranformation_config.data_transformed_train_dir,transformed_test_file_path=self.data_tranformation_config.data_transformed_test_dir)

                              return data_transformation_artifact

                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e