import random
import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from mini import MinioClient

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
