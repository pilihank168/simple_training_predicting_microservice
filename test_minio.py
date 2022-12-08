import pytest
from mini import *

minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')

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
