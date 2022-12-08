from fastapi import HTTPException
from minio import Minio
from minio.error import S3Error
from io import StringIO
import pandas as pd

class MinioClient:
    def __init__(self, ip, port, ac, sc) -> None:
        self.minio = Minio(f"{ip}:{port}", access_key=ac, secret_key=sc, secure=False)

    def get(self, path):
        """get the uploaded file in the minio storage

        Args:
            path: the path in minio storage, in form of '{bucket}/{filename}'

        Returns:
            a file object that can be read by function like pandas.read_csv
        """
        try:
            bucket, obj_name = path.split('/', 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid file path")
        if not self.minio.bucket_exists(bucket):
            raise HTTPException(status_code=400, detail='non exist bucket')
        try:
            self.minio.stat_object(bucket, obj_name)
        except S3Error as err:
            if err.message=='Object does not exist':
                raise HTTPException(status_code=404, detail="File not found in storage")
        try:
            response = self.minio.get_object(bucket, obj_name)
            obj = StringIO(str(response.read(), 'utf-8'))
        finally:
            response.close()
            response.release_conn()
        return obj
