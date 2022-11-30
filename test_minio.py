import pytest
from mini import *

def test_minio_valid():
    assert minio_getter('hello/train_iris.csv')

def test_minio_no_bucket():
    with pytest.raises(HTTPException):
        minio_getter('train_iris.csv')

def test_minio_non_exist():
    with pytest.raises(HTTPException):
        minio_getter('hello/valid_iris.csv')
