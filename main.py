from src.insurance_fraud.logger import logging
from src.insurance_fraud.entity import *
from src.insurance_fraud.pipeline.training_pipeline import TrainPipeline
from src.insurance_fraud.ml.models.model_estimator import ModelResolver
from src.insurance_fraud.utils.main_utils import load_object
from src.insurance_fraud.constant.training_pipeline import *
from src.insurance_fraud.utils.main_utils import extracting_feature,handiling_categorical_features,handiling_outlier
from src.insurance_fraud.constant.training_pipeline.data_transformation_constants import AFTER_EXTRACTION_UNNECESSARY_COLS
from flask import render_template, request, redirect,Flask
import os
import pandas as pd
from flask_cors import CORS, cross_origin
import pickle
app=Flask(__name__,template_folder="templates")



@app.route('/',methods=['POST', 'GET'])
@cross_origin()
def home():
    predict=None
    if request.files:
        training_pipeline=TrainPipeline()
        model_pusher_artifact=training_pipeline.run_pipeline()
        preprocessor=model_pusher_artifact.preprocessor_file_path
        model=model_pusher_artifact.saved_model_path
        uploaded_file = request.files['filename']
        filepath = os.path.join(app.config['FILE_UPLOADS'], uploaded_file.filename)
        df=pd.read_csv(filepath)
        df=handiling_outlier(df,"age_of_car")
        df=handiling_outlier(df,"age_of_policyholder")
        df=df.reset_index(drop=True)
        df=extracting_feature(dataframe=df,old_col_name="max_torque",new_col_name="torque_nm",unit="Nm",pos=0)
        df=extracting_feature(dataframe=df,old_col_name="max_torque",new_col_name="torque_rpm",unit="rpm",pos=1)
        df=extracting_feature(dataframe=df,old_col_name="max_power",new_col_name="power_bhp",unit="bhp",pos=0)
        df=extracting_feature(dataframe=df,old_col_name="max_power",new_col_name="power_rpm",unit="rpm",pos=1)
        df=df.drop(AFTER_EXTRACTION_UNNECESSARY_COLS,axis=1)
        df=handiling_categorical_features(df,col_name="segment")
        df=handiling_categorical_features(df,col_name="model")
        df["is_esc"]=df["is_esc"].map({'Yes':1,'No':0})
        df["is_adjustable_steering"]=df["is_adjustable_steering"].map({'Yes':1,'No':0})
        df["is_parking_sensors"]=df["is_parking_sensors"].map({'Yes':1,'No':0})
        df["is_parking_camera"]=df["is_parking_camera"].map({'Yes':1,'No':0})
        df["transmission_type"]=df["transmission_type"].map({'Manual':1,'Automatic':0})
        df["is_brake_assist"]=df["is_brake_assist"].map({'Yes':1,'No':0})
        df["is_central_locking"]=df["is_central_locking"].map({'Yes':1,'No':0})
        df["is_power_steering"]=df["is_power_steering"].map({'Yes':1,'No':0})
        df=pd.concat([df,pd.get_dummies(df[["segment","model","fuel_type"]],drop_first=True)],axis=1)
        df=df.drop(["segment","model","fuel_type"],axis=1)
        input_feature_train_df=df.drop(columns=["is_claim"],axis=1)
        preprocessor_pickle = pickle.load(open(preprocessor, 'rb'))
        df=pd.DataFrame(preprocessor_pickle.transform(input_feature_train_df),columns=input_feature_train_df.columns)
        model_pickle=pickle.load(open(model,'rb'))
        predict=model_pickle.predict(df)
        df["predicted_result"]=predict
        df.to_csv("F:\\Project Ineuron\\Insurance_Fraud_Prediction\\artifact\\12_03_2022_58_11\\data_ingestion\\ingested\\result.csv",index=False)
    return render_template('index.html',data=predict)

app.config['FILE_UPLOADS'] = "F:\\Project Ineuron\\Insurance_Fraud_Prediction\\artifact\\12_03_2022_58_11\\data_ingestion\\ingested"



if __name__ == '__main__':
   
    app.run(debug=True)