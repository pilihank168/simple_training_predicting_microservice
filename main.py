from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from io import StringIO
import numpy as np
from sklearn import tree
#import skl2onnx
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from mongo import *
import random
import pandas as pd

app = FastAPI()

@app.get("/model") 
async def testing(model_id: str, data_path: str):
#    model = onnx.InferenceSession(f'{model_id}.onnx')
#    input_name = sess.get_inputs()[0].name
#    label_name = sess.get_outputs()[0].name
#    preds = model.run([label_name], {input_name: x.astype(numpy.float32)})
    model_file, features = query_model(model_id)
    x, y, _ = data2xy(data_path, features)
    with open('tmp.joblib', 'wb') as f:
        f.write(model_file)
    model = load(f'tmp.joblib')
    preds = model.predict(x)
    print(preds, features)
    acc = model.score(x, y)
    print(acc)
    return {"acc": acc, "preds": preds.tolist()}

@app.post("/model")
async def training(data_path: str):
    x, y, features = data2xy(data_path)
    model = tree.DecisionTreeClassifier()
    cv_score = cross_val_score(model, x, y)
    model = model.fit(x, y)
#    onnx_model = skl2onnx.to_onnx(model, x[:1])
    dump(model, 'tmp.joblib')
    with open('tmp.joblib', 'rb') as model_file:
        model_id = insert_model(score=cv_score.mean(), features=features, model=model_file)
    print(model_id, features)
#    with open(f'{model_id}.onnx', 'wb') as f:
#        f.write(onnx_model.SerializeToString())
    return {"model_id": model_id, 'cv_score': cv_score.mean()}

def data2xy(data_path, features=None):
    try:
        df = pd.read_csv(data_path)
    except Exception:
        raise HTTPException(status_code=406, detail="bad file")
    if features==None:
        feats = df.columns.tolist()[:-1]
        random.shuffle(feats)
        features = feats[:len(feats)//2]
    else:
        for feat in features:
            if feat not in df:
                raise HTTPException(status_code=422, detail=f"Missing Features: {feat}")
    x = df[features].to_numpy()
    y = df['label'].to_numpy()
    return x, y, features

@app.get("/models")
async def list_models():
    return {'models':list_all()}
app.mount("/", StaticFiles(directory="static"), name="static")
