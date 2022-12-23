from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from utils import training_flow, testing_flow
from pydantic_model import *
from mongo import MongoClient
from prefect.deployments import Deployment, run_deployment
from prefect.client.orion import get_client

import pytest

app = FastAPI()

mongo_client = MongoClient('mongodb')
mongo_config = ['mongodb', 'hello_db']

train_deployment = Deployment.build_from_flow(flow=training_flow, name='train', version=1, apply=True)
test_deployment = Deployment.build_from_flow(flow=testing_flow, name='test', version=1, apply=True)

@pytest.mark.skip
@app.post("/testing", response_model=FlowRun) 
async def testing(body: TestRequest) -> FlowRun:
    """run testing flow on prefect deployment

    Args:
        body (TestRequest): request body consists of model_id(str) and data_path(str)
            model_id (str): id to query the model
            data_path (str): path to testing data

    Returns:
        flow_run_id: the id that can be used to track flow progress, this id is also the id in mongodb
    """
    flow_run = await run_deployment(name="testing-flow/test", parameters={"model_id":body.model_id, "data_path":body.data_path, "mongo_config":mongo_config}, timeout=0)
    return FlowRun.parse_obj({'flow_run_id':str(flow_run.id)})

@app.post("/training", response_model=FlowRun)
async def training(body: TrainRequest) -> FlowRun:
    """run training flow on prefect deployment. 

    Args:
        body (TrainRequest): request body

    Returns:
        flow_run_id: the id that can be used to track flow progress, this id is also the id in mongodb
    """
    flow_run = await run_deployment(name="training-flow/train", parameters={"body":body, "mongo_config":mongo_config}, timeout=0)
    return FlowRun.parse_obj({'flow_run_id': str(flow_run.id)})

@app.get("/training/{training_id}", response_model=TrainResponse)
async def query_training(training_id) -> TrainResponse:
    """query_training.

    Args:
        training_id: id of training flow run

    Returns:
        state: one of 'completed', 'failed', 'running'
        failed_step/running_step: the name of failed task or currently running task
        result: if the flow run is successfully completed, the scores of cross validation is put here
    """
    client = get_client()
    graph = await client._client.get(f'/flow_runs/{training_id}/graph')
    flow_run = await client.read_flow_run(training_id)
    current_task = ['still pending', 'load data', 'preprocess data', 'train model', 'save model'][len(graph.json())]
    if flow_run.state.is_final():
        if flow_run.state.is_completed():
            return TrainResponse.parse_obj({'state': 'completed', 'result': TrainResult(cv_scores=Scores.parse_obj(mongo_client.query_training(training_id)))})
        else:
            return TrainResponse.parse_obj({'state':'failed', 'failed_step':current_task, 'failed_message':flow_run.state.message})
    else:
        return TrainResponse.parse_obj({'state': 'running', 'running_step': current_task})

@app.get("/testing/{testing_id}", response_model=TestResponse)
async def query_testing(testing_id) -> TestResponse:
    """query_testing.

    Args:
        testing_id: id of testing flow run

    Returns:
        state: one of 'completed', 'failed', 'running'
        failed_step/running_step: the name of failed task or currently running task
        result: if the flow run is successfully completed, 
            scores: the evaluated scores corresponds to those used for cross validation when training
            preds: the predicted labels, in the same order of the csv file
    """
    client = get_client()
    graph = await client._client.get(f'/flow_runs/{testing_id}/graph')
    flow_run = await client.read_flow_run(testing_id)
    current_task = ['still pending', 'load model', 'load data', 'preprocess data', 'test model'][len(graph.json())]
    if flow_run.state.is_final():
        if flow_run.state.is_completed():
            return TestResponse.parse_obj({'state': 'completed', 'result': TestResult.parse_obj(mongo_client.query_testing(testing_id))})
        else:
            return TestResponse.parse_obj({'state':'failed', 'failed_step':current_task, 'failed_message':flow_run.state.message})
    else:
        return TestResponse.parse_obj({'state': 'running', 'running_step': current_task})
