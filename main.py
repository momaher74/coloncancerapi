from pydantic import BaseModel, conlist
import pandas as pd
from fastapi import FastAPI, Body, UploadFile, Header, File
from joblib import load
import uvicorn


app = FastAPI(title="Colon ML API", description="Colon for iris dataset ml model", version="1.0", )



@app.on_event('startup')
def load_model():
    global model
    model = load('lib/Colon_cancer_svc.joblib')


@app.post("/uploadcsv")
async def upload_file(file: UploadFile):
    dataframe = pd.read_csv(file.file)
    prediction = model.predict(dataframe).tolist()
    result = pd.DataFrame(prediction)
    result = result.replace({0: 'normal', 1: 'adenocarcinoma'})
    result = result.values.tolist()
    return {"prediction": result, }



uvicorn.run(app=app, port=5000, log_level="info", host="0.0.0.0")


