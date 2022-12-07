from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import numpy as np
from sklearn import tree
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from mongo import *
import random
import pandas as pd
from pydantic_model import *
from mini import *
import pytest
app = FastAPI()

minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')

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
    model_file, features, label, matrix = query_model(model_id)
    x, y, _ = data2xy(data_path, label, features)
    with open('tmp.joblib', 'wb') as f:
        f.write(model_file)
    model = load(f'tmp.joblib')
    preds = model.predict(x)
    scores = {mat:scorer(y, preds, mat) for mat in matrix}
    return TestResponse(scores=scores, preds=preds.tolist())
#    return {"scores": scores, "preds": preds.tolist()}

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
        model_id = insert_model(cv_scores, features, model_file, body.target_name, body.eval_matrix, body.num_cv_fold, body.dtree_param.dict())
    #return {"model_id": model_id, 'cv_scores': cv_scores}
    return TrainResponse(model_id=model_id, cv_scores=cv_scores)

def scorer(y, p, matrix: str) -> float:
    """scorer.

    Args:
        y: the ground-truth labels
        p: the predicted labels
        matrix (str): string that specifies the matrix used for model evaluation

    Returns:
        float: the evaluation score
    """
    if matrix=='accuracy':
        return accuracy_score(y, p)
    elif matrix=='precision_micro':
        return precision_score(y, p, average='micro')
    elif matrix=='precision_macro':
        return precision_score(y, p, average='macro')
    elif matrix=='recall_micro':
        return recall_score(y, p, average='micro')
    elif matrix=='recall_macro':
        return recall_score(y, p, average='macro')
    elif matrix=='f1_micro':
        return f1_score(y, p, average='micro')
    elif matrix=='f1_macro':
        return f1_score(y, p, average='macro')
    elif matrix=='neg_log_loss':
        return -log_loss(y, p)
    raise HTTPException(status_code=422, detail='unknown matrix for evaluation')

def data2xy(data_path, target_name, features=None):
    """Read the data and split it into X and y

    Args:
        data_path: the path of training/testing data (.csv)
        target_name: the name of label column
        features: the columns that are used for training/testing, randomly selected when it is None (for training)
    """
    try:
        df = pd.read_csv(minio_client.get(data_path))
    except HTTPException as err:
        raise err
    except Exception:
        raise HTTPException(status_code=406, detail="bad file")
    # check if target column exists in training stage (it's acceptable in testing)
    if target_name not in df:
        if features==None:
            raise HTTPException(status_code=422, detail=f"Target Name {target_name} doesn't exist")
        else:
            y = None
    else:
        if df[target_name].isnull().values.any():
            raise HTTPException(status_code=406, detail="Nan value in target column")
        y = df[target_name].to_numpy()
    # randomly select some features for training
    if features==None:
        feats = [feat for feat in df.columns.tolist() if feat!=target_name]
        random.shuffle(feats)
        features = feats[:len(feats)//2]
    else:
        for feat in features:
            if feat not in df:
                raise HTTPException(status_code=422, detail=f"Missing Features: {feat}")
    if df[features].isnull().values.any():
        raise HTTPException(status_code=406, detail="Nan value in data")
    x = df[features].to_numpy()
    return x, y, features

@app.get("/models")
async def list_models():
    return {'models':list_all()}
app.mount("/", StaticFiles(directory="static"), name="static")
