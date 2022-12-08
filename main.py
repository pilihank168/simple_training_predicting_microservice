import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sklearn import tree
from joblib import dump, load
from sklearn.model_selection import cross_val_score

from utils import scorer, data2xy
from pydantic_model import *
from mongo import MongoClient

import pytest

app = FastAPI()

mongo_client = MongoClient('mongodb')

@pytest.mark.skip
@app.get("/model") 
def testing(model_id: str, data_path: str) -> TestResponse:
    """testing.

    Args:
        model_id (str): id to query the model
        data_path (str): path to testing data

    Returns:
        TestResponse:
            scores: the evaluated scores corresponds to those used for cross validation when training
            preds: the predicted labels, in the same order of the csv file
    """
    model_file, features, label, matrix = mongo_client.query_model(model_id)
    x, y, _ = data2xy(data_path, label, features)
    with open('tmp.joblib', 'wb') as f:
        f.write(model_file)
    model = load(f'tmp.joblib')
    preds = model.predict(x)
    scores = {mat:scorer(y, preds, mat) for mat in matrix}
    return TestResponse(scores=scores, preds=preds.tolist())

@app.post("/model")
def training(body: TrainRequest) -> TrainResponse:
    """training.

    Args:
        body (TrainRequest): request body

    Returns:
        TrainResponse: model_id to query this training in mongo db, and the scores of cross validation
    """
    x, y, features = data2xy(body.data_path, body.target_name)
    model = tree.DecisionTreeClassifier(**body.dtree_param.dict())
    cv_scores = {mat: np.nanmean(cross_val_score(model, x, y, scoring=mat, cv=body.num_cv_fold)) for mat in body.eval_matrix}
    model = model.fit(x, y)
    dump(model, 'tmp.joblib')
    with open('tmp.joblib', 'rb') as model_file:
        model_id = mongo_client.insert_model(cv_scores, features, model_file, body.target_name, body.eval_matrix, body.num_cv_fold, body.dtree_param.dict())
    return TrainResponse(model_id=model_id, cv_scores=cv_scores)

@app.get("/models")
async def list_models():
    return {'models':list_all()}
app.mount("/", StaticFiles(directory="static"), name="static")
