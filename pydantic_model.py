from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, ValidationError, validator, Extra, root_validator

def valid_data_path(path: str) -> str:
    """valid_data_path checker, as validator used by TrainRequest and TestRequest

    Args:
        path (str): data path to be checked if it is valid

    Returns:
        str: the valid path itself. If data path is not valid, ValidationError will be raised.
    """
    if path[-4:] in ['.csv', '.txt']:
        return path
    raise ValidationError('invalid filename extension')

class DecisionTreeParam(BaseModel):
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'
    splitter: Literal['best', 'random'] = 'best'
    class Config:
        extra = Extra.forbid
        error_msg_templates = {'value_error.extra': 'Invalid attribute name'}

class Scores(BaseModel):
    accuracy: Union[float, None]=None
    recall_micro: Union[float, None]=None
    recall_macro: Union[float, None]=None
    precision_micro: Union[float, None]=None
    precision_macro: Union[float, None]=None
    f1_micro: Union[float, None]=None
    f1_macro: Union[float, None]=None
    neg_log_loss: Union[float, None]=None

    @root_validator
    def non_empty_scores(cls, v):
        if not any(v.values()):
            raise ValidationError('cv scores cannot be empty')
        return v

    class Config:
        extra = Extra.forbid
        error_msg_templates = {'value_error.extra': 'Invalid evaluation matrix'}

class TrainRequest(BaseModel):
    data_path: str
    target_name: str = 'label'
    eval_matrix: list[str] = ['accuracy']
    num_cv_fold: int = 5
    dtree_param: DecisionTreeParam = DecisionTreeParam()

    @validator('eval_matrix')
    def valid_matrix_checking(cls, matrix):
        for mat in matrix:
            if mat not in ['accuracy', 'recall_micro', 'recall_macro', 'precision_micro',
                            'precision_macro', 'f1_macro', 'f1_micro', 'neg_log_loss']:
                raise ValidationError('invalid evaluation matrix')
        return matrix
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)

class TestRequest(BaseModel):
    __test__= False
    data_path: str
    model_id: str
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)

class TrainResult(BaseModel):
    cv_scores: Scores

class TrainResponse(BaseModel):
    state: Literal['completed', 'failed', 'running']
    failed_step: Literal[None, 'still pending', 'load data', 'preprocess data', 'train model', 'save model'] = None
    running_step: Literal[None, 'still pending', 'load data', 'preprocess data', 'train model', 'save model'] = None
    failed_message: Union[str, None] = None
    result: Union[TrainResult, None] = None
    
    @root_validator
    def state_corresponding_fields(cls, values):
        state = values.get('state')
        nones = [values.get(name)==None for name in ['failed_step', 'failed_message', 'running_step', 'result']]
        if state=='completed' and (not all(nones[:3]) or nones[3]):
            raise ValidationError('Fields does not correspond to state completed')
        elif state=='failed' and (any(nones[:2]) or not all(nones[2:])):
            raise ValidationError('Fields does not correspond to state failed')
        elif state=='running' and (nones[2] or not all(nones[:2]+nones[3:])):
            raise ValidationError('Fields does not correspond to state running')
        return values

class TestResult(BaseModel):
    __test__= False
    scores: Scores
    preds: list[int]

class TestResponse(BaseModel):
    __test__= False
    state: Literal['completed', 'failed', 'running']
    failed_step: Literal[None, 'still pending', 'load model', 'load data', 'preprocess data', 'test model'] = None
    running_step: Literal[None, 'still pending', 'load model', 'load data', 'preprocess data', 'test model'] = None
    failed_message: Union[str, None] = None
    result: Union[TestResult, None] = None
    
    @root_validator
    def state_corresponding_fields(cls, values):
        state = values.get('state')
        nones = [values.get(name)==None for name in ['failed_step', 'failed_message', 'running_step', 'result']]
        if state=='completed' and (not all(nones[:3]) or nones[3]):
            raise ValidationError('Fields does not correspond to state completed')
        elif state=='failed' and (any(nones[:2]) or not all(nones[2:])):
            raise ValidationError('Fields does not correspond to state failed')
        elif state=='running' and (nones[2] or not all(nones[:2]+nones[3:])):
            raise ValidationError('Fields does not correspond to state running')
        return values

class FlowRun(BaseModel):
    flow_run_id: str
