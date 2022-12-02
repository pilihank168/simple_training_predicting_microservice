import pytest
from main import *

def count_db():
    def count(collection):
        return len(list(collection.find({})))
    l0 = count(mongo_client.hello_db.models)
    l1 = count(mongo_client.hello_db.fs.files)
    l2 = count(mongo_client.hello_db.fs.chunks)
    return l0, l1, l2

def test_db_valid_training():
    lens_0 = count_db()
    training(TrainRequest(data_path='hello/train_iris.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]-1

def test_db_failed_training():
    lens_0 = count_db()
    with pytest.raises(HTTPException):
        training(TrainRequest(data_path='hello/train_iris.csv', target_name='labe1', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))
    lens_1 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_testing():
    model_id = mongo_client.hello_db.models.find_one({})['_id']
    lens_0 = count_db()
    testing(model_id=str(model_id), data_path='hello/test_iris.csv')
    lens_1 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]

def test_db_failed_testing():
    model_id = mongo_client.hello_db.models.find_one({})['_id']
    lens_0 = count_db()
    with pytest.raises(HTTPException):
        testing(model_id=str(model_id), data_path='hello/test_iris.txt')
    lens_1 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
