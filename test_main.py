import pytest
from main import *
from mini import *

@pytest.fixture(scope="session", autouse=True)
def clear_db(request):
    def cleanup_db():
        mongo_client.hello_db.models.delete_many({})
        mongo_client.hello_db.fs.files.delete_many({})
        mongo_client.hello_db.fs.chunks.delete_many({})
    request.addfinalizer(cleanup_db)

def count_db():
    count = lambda col:len(list(col.find({})))
    l0 = count(mongo_client.hello_db.models)
    l1 = count(mongo_client.hello_db.fs.files)
    l2 = count(mongo_client.hello_db.fs.chunks)
    return l0, l1, l2

def test_db_valid_training():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)==0
    training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and max(lens_1)==1

def test_db_failed_training():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    with pytest.raises(HTTPException):
        training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='labe1', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_testing():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = mongo_client.hello_db.models.find_one({})['_id']
    testing(model_id=str(model_id), data_path='hello-docker/test_iris.csv')
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_failed_testing():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = mongo_client.hello_db.models.find_one({})['_id']
    with pytest.raises(HTTPException):
        testing(model_id=str(model_id), data_path='hello-docker/test_iris.txt')
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
