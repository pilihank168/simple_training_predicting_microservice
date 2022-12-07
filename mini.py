from fastapi import HTTPException
from minio import Minio
from minio.error import S3Error
from io import StringIO
import pandas as pd

#minio_client = Minio('127.0.0.1:9000', access_key='eqjPdZZB9ZZTOW4U', secret_key='5xMedNSLErRSia7DnUaeVBkeX6W6fLUo', secure=False)
#minio_client = Minio('minio:9000', access_key='Mhn9NcXrvOcfZnAI', secret_key='uDorPlk7wb7tp7SUWx8vO288BHFuLURd', secure=False)

class MinioClient:
    def __init__(self, ip, port, ac, sc) -> None:
        self.minio = Minio(f"{ip}:{port}", access_key=ac, secret_key=sc, secure=False)

    def get(self, path):
        """get.

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

minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')
