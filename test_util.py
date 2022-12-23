import pytest
from utils import load_data, process_data, scorer
from fastapi import HTTPException

# data2xy
def test_data2xy_valid_train():
    df = load_data.fn('hello-docker/train_iris.csv')
    assert process_data.fn(df, 'label')

def test_data2xy_valid_test():
    df = load_data.fn('hello-docker/train_iris.csv')
    assert process_data.fn(df, 'label', ['feat1', 'feat3'])

def test_data2xy_valid_test_without_target():
    df = load_data.fn('hello-docker/train_iris.csv')
    assert process_data.fn(df, 'labe1', ['feat2', 'feat4'])

def test_data2xy_bad_file():
    with pytest.raises(HTTPException):
        df = load_data.fn('hello-docker/main.py')

def test_data2xy_invalid_target_name():
    df = load_data.fn('hello-docker/train_iris.csv')
    with pytest.raises(HTTPException):
        process_data.fn(df, 'labe1')

def test_data2xy_invalid_feature_name():
    df = load_data.fn('hello-docker/train_iris.csv')
    with pytest.raises(HTTPException):
        process_data.fn(df, 'label', 'bad_feat')

def test_data2xy_nan_in_target():
    df = load_data.fn('hello-docker/test_iris.txt')
    with pytest.raises(HTTPException):
        process_data.fn(df, 'label', ['feat1', 'feat3'])

def test_data2xy_nan_in_feature():
    df = load_data.fn('hello-docker/train_iris.txt')
    with pytest.raises(HTTPException):
        process_data.fn(df, 'label', ['feat1', 'feat2'])

def test_data2xy_nan_in_other_column():
    df = load_data.fn('hello-docker/train_iris.txt')
    assert process_data.fn(df, 'label', ['feat4', 'feat3'])

# scorer
def test_scorer_valid():
    assert scorer([0, 1, 2], [2, 1, 0], 'accuracy')==1/3

def test_scorer_unknown_matrix():
    with pytest.raises(HTTPException):
        scorer([0, 1, 2], [2, 1, 0], 'unknown_matrix')
