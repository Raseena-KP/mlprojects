from flask import Flask,request,render_template,url_for
from src.exception import CustomException
import pandas as pd
import numpy as np
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData,Predict_pipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
        
        df=data.convert_input_to_dataframe()
        print(df)
        print('before prediction')

        predict_pipeline_obj=Predict_pipeline()
        result=predict_pipeline_obj.predict(df)
        print(result)
        return render_template ('home.html',result=result[0])

if __name__=="__main__":
    app.run(port=8080)
