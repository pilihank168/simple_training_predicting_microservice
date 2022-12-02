from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, ValidationError, validator, Extra, root_validator

def valid_data_path(path: str) -> str:
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

class TrainResponse(BaseModel):
    model_id: str
    cv_scores: Scores

class TestRequest(BaseModel):
    __test__=False
    data_path: str
    model_id: str
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)

class TestResponse(BaseModel):
    __test__=False
    scores: Scores
    preds: list[int]
