import pytest
from fastapi import HTTPException
import main#from main import training, testing
from mongo import MongoClient
from mini import MinioClient
from pydantic_model import *

main.mongo_client = MongoClient('mongodb', db_name="test")
#minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')

@pytest.fixture(scope="session", autouse=True)
def clear_db(request):
    """the cleanup function executed after this session ends to empty the db for testing
    """
    def cleanup_db():
        main.mongo_client.db.models.delete_many({})
        main.mongo_client.db.fs.files.delete_many({})
        main.mongo_client.db.fs.chunks.delete_many({})
    request.addfinalizer(cleanup_db)

def count_db():
    count = lambda col:len(list(col.find({})))
    l0 = count(main.mongo_client.db.models)
    l1 = count(main.mongo_client.db.fs.files)
    l2 = count(main.mongo_client.db.fs.chunks)
    return l0, l1, l2

def test_db_valid_training():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)==0
    main.training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and max(lens_1)==1

def test_db_failed_training():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    with pytest.raises(HTTPException):
        main.training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='labe1', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_testing():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = main.mongo_client.db.models.find_one({})['_id']
    main.testing(model_id=str(model_id), data_path='hello-docker/test_iris.csv')
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_failed_testing():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = main.mongo_client.db.models.find_one({})['_id']
    with pytest.raises(HTTPException):
        main.testing(model_id=str(model_id), data_path='hello-docker/test_iris.txt')
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
