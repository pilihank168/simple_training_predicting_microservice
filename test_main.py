import pytest
from fastapi import HTTPException
import main#from main import training, testing
from mongo import MongoClient
from mini import MinioClient
from pydantic_model import *
import asyncio
from prefect.client.orion import get_client#OrionClient
import time

main.mongo_client = MongoClient('mongodb', db_name="test")
main.mongo_config = ['mongodb', 'test']
#minio_client = MinioClient('minio', '9000', 'Mhn9NcXrvOcfZnAI', 'uDorPlk7wb7tp7SUWx8vO288BHFuLURd')

@pytest.fixture(scope="session", autouse=True)
def clear_db(request):
    """the cleanup function executed after this session ends to empty the db for testing
    """
    def cleanup_db():
        main.mongo_client.db.models.delete_many({})
        main.mongo_client.db.fs.files.delete_many({})
        main.mongo_client.db.fs.chunks.delete_many({})
    cleanup_db()
    request.addfinalizer(cleanup_db)

def count_db():
    count = lambda col:len(list(col.find({})))
    l0 = count(main.mongo_client.db.models)
    l1 = count(main.mongo_client.db.fs.files)
    l2 = count(main.mongo_client.db.fs.chunks)
    return l0, l1, l2

async def get_final_state(flow_run_id):
    client = get_client()
    flow_run = await client.read_flow_run(flow_run_id)
    while not flow_run.state.is_final():
        time.sleep(2)
        flow_run = await client.read_flow_run(flow_run_id)
    return flow_run.state

def test_training_valid():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)==0
    flow_run_id = asyncio.run(main.training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))).flow_run_id
    final_state = asyncio.run(get_final_state(flow_run_id))
    assert final_state.is_completed()
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and max(lens_1)==1
    result = asyncio.run(main.query_training(str(flow_run_id)))
    assert result.state=='completed'

def test_training_failed_load_data():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    #with pytest.raises(HTTPException):
    flow_run_id = asyncio.run(main.training(TrainRequest(data_path='hello-docker/valid_iris.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))).flow_run_id
    final_state = asyncio.run(get_final_state(flow_run_id))
    assert final_state.is_failed() and "HTTPException" in final_state.message
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
    result = asyncio.run(main.query_training(flow_run_id))
    assert result.state=='failed' and result.failed_step=='load data'

def test_training_failed_process_data():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    #with pytest.raises(HTTPException):
    flow_run_id = asyncio.run(main.training(TrainRequest(data_path='hello-docker/train_iris.csv', target_name='labe1', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random')))).flow_run_id
    final_state = asyncio.run(get_final_state(flow_run_id))
    assert final_state.is_failed() and "HTTPException" in final_state.message
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
    result = asyncio.run(main.query_training(flow_run_id))
    assert result.state=='failed' and result.failed_step=='preprocess data'

def test_testing():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = main.mongo_client.db.models.find_one({})['_id']
    flow_run_id = asyncio.run(main.testing(TestRequest(model_id=str(model_id), data_path='hello-docker/test_iris.csv'))).flow_run_id
    final_state = asyncio.run(get_final_state(flow_run_id))
    assert final_state.is_completed()
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
    result = asyncio.run(main.query_testing(str(flow_run_id)))
    assert result.state=='completed'

def test_testing_failed_process_data():
    lens_0 = count_db()
    assert max(lens_0)==min(lens_0) and max(lens_0)>0
    model_id = main.mongo_client.db.models.find_one({})['_id']
    #with pytest.raises(HTTPException):
    flow_run_id = asyncio.run(main.testing(TestRequest(model_id=str(model_id), data_path='hello-docker/test_iris.txt'))).flow_run_id
    final_state = asyncio.run(get_final_state(flow_run_id))
    assert final_state.is_failed() and "HTTPException" in final_state.message
    lens_1 = count_db()
    assert max(lens_1)==min(lens_1) and lens_0[0]==lens_1[0]
    result = asyncio.run(main.query_testing(flow_run_id))
    assert result.state=='failed' and result.failed_step=='preprocess data'

