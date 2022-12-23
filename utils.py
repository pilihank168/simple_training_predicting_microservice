import random
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from mini import MinioClient
from prefect import flow, task
from prefect.context import get_run_context
from sklearn import tree
from joblib import dump, load
from sklearn.model_selection import cross_val_score
from mongo import MongoClient
from pydantic_model import *

minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')

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

@task
def load_data(data_path):
    """Load data from minio storage

    Args:
        data_path: the path of training/testing data (.csv)
    """
    try:
        df = pd.read_csv(minio_client.get(data_path))
    except HTTPException as err:
        raise err
    except Exception:
        raise HTTPException(status_code=406, detail="bad file")
    return df

@task
def process_data(df, target_name, features=None):
    """Split data into X and y according to column names

    Args:
        df: loaded dataframe
        target_name: the name of label column
        features: the columns that are used for training/testing, randomly selected when it is None (for training)
    """
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

@task
def train_model(x, y, body):
    model = tree.DecisionTreeClassifier(**body.dtree_param.dict())
    cv_scores = {mat: np.nanmean(cross_val_score(model, x, y, scoring=mat, cv=body.num_cv_fold)) for mat in body.eval_matrix}
    model = model.fit(x, y)
    return model, cv_scores

@task
def test_model(model, x, y, matrix):
    preds = model.predict(x)
    scores = {mat:scorer(y, preds, mat) for mat in matrix}
    return preds, scores

@task
def save_model(model, cv_scores, features, body, mongo_client, model_id=None):
    """save model and training information by mongo db client, the id of this entry is same as flow run
    """
    dump(model, 'tmp.joblib')
    with open('tmp.joblib', 'rb') as model_file:
        model_id = mongo_client.insert_model(cv_scores, features, model_file, body.target_name, body.eval_matrix, body.num_cv_fold, body.dtree_param.dict(), model_id)
    return model_id

@task
def load_model(model_id, mongo_client):
    model_file, features, label, matrix = mongo_client.query_model(model_id)
    with open('tmp.joblib', 'wb') as f:
        f.write(model_file)
    model = load(f'tmp.joblib')
    return model, features, label, matrix

@flow
def training_flow(body, mongo_config):
    mongo_client = MongoClient(mongo_config[0], db_name=mongo_config[1])
    body = TrainRequest.parse_obj(body)
    df = load_data(body.data_path)
    x, y, features = process_data(df, body.target_name)
    model, cv_scores = train_model(x, y, body)
    model_id = save_model(model, cv_scores, features, body, mongo_client, model_id=str(get_run_context().flow_run.id))
    return model_id, cv_scores

@flow
def testing_flow(model_id, data_path, mongo_config):
    mongo_client = MongoClient(mongo_config[0], db_name=mongo_config[1])
    model, features, label, matrix = load_model(model_id, mongo_client)
    df = load_data(data_path)
    x, y, _ = process_data(df, label, features)
    preds, scores = test_model(model, x, y, matrix)
    testing_id = str(get_run_context().flow_run.id)
    mongo_client.insert_testing(str(testing_id), preds, scores)
    return preds, scores
