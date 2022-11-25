from typing import Dict, List, Optional
from pydantic import BaseModel, ValidationError, validator

def valid_data_path(path: str) -> str:
    if path[-4:] in ['.csv', '.txt']:
        return path
    raise ValidationError('invalid filename extension')

class DecisionTreeParam(BaseModel):
    criterion: str = 'gini'
    splitter: str = 'best'
    @validator('criterion')
    def valid_criterion(cls, v):
        if v not in ['gini', 'entropy', 'log_loss']:
            raise ValidationError(f'invalid parameter {v} for decision tree classifier\'s critierion')
        return v
    @validator('splitter')
    def valid_splitter(cls, v):
        if v not in ['best', 'random']:
            raise ValidationError(f'invalid parameter {v} for decision tree classifier\'s splitter')
        return v

class TrainRequest(BaseModel):
    data_path: str
    target_name: str = 'label'
    eval_matrix: list[str] = ['accuracy']
    num_cv_fold: int = 5
    dtree_param: DecisionTreeParam

    @validator('eval_matrix')
    def valid_matrix_checking(cls, matrix):
        for mat in matrix:
            if mat not in ['accuracy', 'recall', 'precision', 'f1_macro', 'f1_micro', 'log_loss', 'roc_auc']:
                raise ValidationError('invalid evaluation matrix')
        return matrix
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)

class TestRequest(BaseModel):
    data_path: str
    model_id: str
    _valid_path = validator('data_path', allow_reuse=True)(valid_data_path)
