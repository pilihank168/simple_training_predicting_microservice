from fastapi import HTTPException
from minio import Minio
from minio.error import S3Error
from io import StringIO
import pandas as pd

def minio_getter(path):
    """minio_getter.

    Args:
        path: the path in minio storage, in form of '{bucket}/{filename}'

    Returns:
        a file object that can be opened by function like pandas.read_csv
    """
    try:
        bucket, obj_name = path.split('/', 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid file path")
    try:
        _ = minio_client.stat_object(bucket, obj_name)
    except S3Error as err:
        if err.message=='Object does not exist':
            raise HTTPException(status_code=404, detail="File not found in storage")
    try:
        response = minio_client.get_object(bucket, obj_name)
        obj = StringIO(str(response.read(), 'utf-8'))
    finally:
        response.close()
        response.release_conn()
    return obj

minio_client = Minio('127.0.0.1:9000', access_key='eqjPdZZB9ZZTOW4U', secret_key='5xMedNSLErRSia7DnUaeVBkeX6W6fLUo', secure=False)
