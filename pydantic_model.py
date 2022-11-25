from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, ValidationError, validator

def valid_data_path(path: str) -> str:
    if path[-4:] in ['.csv', '.txt']:
        return path
    raise ValidationError('invalid filename extension')

class DecisionTreeParam(BaseModel):
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'
    splitter: Literal['best', 'random'] = 'best'

class TrainRequest(BaseModel):
    data_path: str
    target_name: str = 'label'
    eval_matrix: list[str] = ['accuracy']
    num_cv_fold: int = 5
    dtree_param: DecisionTreeParam

    @validator('eval_matrix')
    def valid_matrix_checking(cls, matrix):
        for mat in matrix:
            if mat not in ['accuracy', 'recall_micro', 'recall_macro', 'precision_micro',
                            'precision_macro', 'f1_macro', 'f1_micro', 'neg_log_loss']:
                raise ValidationError('invalid evaluation matrix')
        return matrix
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)

class TestRequest(BaseModel):
    data_path: str
    model_id: str
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)
