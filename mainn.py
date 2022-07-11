# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:38:33 2022

@author: HP
"""

from typing import Optional
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
app = FastAPI()


vector = load(r"D:\api_article\cv.joblib")
model = load(r"D:\api_article\modell.joblib")

class get_usertext(BaseModel):
    usertext :str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(gr:get_usertext):
    text = [gr.usertext]
    vec = vector.transform(text)
    prediction = model.predict(vec)
    prediction = int(prediction)
    if prediction <=0:
        prediction="من الممكن انك تعاني من الاكتئاب"
    elif prediction > 0 and prediction<= 1:
            prediction="من الممكن انك تعاني من اضطراب القلق"
    elif prediction > 1 and prediction<= 2:
            prediction="من الممكن انك تعاني من ميول انتحاريه"
    else:
        prediction = "أخري"

    return {"sentence":gr.usertext,"prediction":prediction}

#uvicorn main:app --reload