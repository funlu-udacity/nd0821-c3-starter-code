# Put the code for your API here.
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from Census import Census
from starter.ml.model import inference
from starter.ml.data import process_data

import pickle
import pandas as pd
import numpy as np
import joblib


#Getting help from https://knowledge.udacity.com/questions/783987
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -df s3-bucket s3://storcensusproject")
    print("AWS set up")
    dvc_output = subprocess.run(
        ["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")



app = FastAPI()
#loading the models
model = pickle.load(open(os.path.join(os.getcwd(), "model", "rf_model.pkl"), 'rb'))
encoder = pickle.load(open(os.path.join(os.getcwd(), "model", "encoder.pkl"), 'rb'))
lb = pickle.load(open(os.path.join(os.getcwd(), "model", "lb.pkl"), 'rb'))

@app.get("/")
async def main():
    return {"Welcome to Udacity Project 3. This app predicts the income above or lower than 50K."}


@app.post("/predict")
def predict_salary(data:Census):

    input_data = data.dict()

    age = input_data["age"]
    workclass = input_data["workclass"]
    fnlgt = input_data["fnlgt"]
    education = input_data["education"]
    education_num = input_data["education_num"]
    marital_status = input_data["marital_status"]
    occupation = input_data["occupation"]
    relationship = input_data["relationship"]
    race = input_data["race"]
    sex = input_data["sex"]
    capital_gain = input_data["capital_gain"]
    capital_loss = input_data["capital_loss"]
    hours_per_week = input_data["hours_per_week"]
    native_country = input_data["native_country"]

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    input_array = np.array([[
        age,
        workclass,
        fnlgt,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        sex,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country
    ]])

    df_input = pd.DataFrame(data=input_array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country"
    ])
    input_x, _, _, _ = process_data(
        df_input, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    prediction = inference(model, input_x)
    if prediction[0]:
        prediction = "Predicted salary is greater than 50K."
    else:
        prediction = "Predicted salary is less than 50K."

    return prediction