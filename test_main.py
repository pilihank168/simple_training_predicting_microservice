import pytest
from main import *
# data2xy
def test_data2xy_valid_train():
    assert data2xy('hello/train_iris.csv', 'label')

def test_data2xy_valid_test():
    assert data2xy('hello/train_iris.csv', 'label', ['feat1', 'feat3'])

def test_data2xy_valid_test_without_target():
    assert data2xy('hello/train_iris.csv', 'labe1', ['feat2', 'feat4'])

def test_data2xy_bad_file():
    with pytest.raises(HTTPException):
        data2xy('hello/main.py', 'label')

def test_data2xy_bad_target():
    with pytest.raises(HTTPException):
        data2xy('hello/train_iris.csv', 'labe1')

def test_data2xy_bad_feature():
    with pytest.raises(HTTPException):
        data2xy('hello/train_iris.csv', 'label', 'bad_feat')

# scorer
def test_scorer_valid():
    assert scorer([0, 1, 2], [2, 1, 0], 'accuracy')

def test_scorer_unkonw_matrix():
    with pytest.raises(HTTPException):
        scorer([0, 1, 2], [2, 1, 0], 'xd')
