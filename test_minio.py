import pytest
from mini import *

def test_minio_valid():
    assert minio_client.get('hello-docker/train_iris.csv')

def test_minio_no_bucket():
    with pytest.raises(HTTPException):
        minio_client.get('train_iris.csv')

def test_minio_non_exist_file():
    with pytest.raises(HTTPException):
        minio_client.get('hello-docker/valid_iris.csv')

def test_minio_non_exist_bucket():
    with pytest.raises(HTTPException):
        minio_client.get('hello_docker/train_iris.csv')
