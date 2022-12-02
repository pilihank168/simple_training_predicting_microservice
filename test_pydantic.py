import pytest
from pydantic_model import *

# train request
def test_training_request_valid():
    assert TrainRequest(data_path='valid_path.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='entropy', splitter='random'))

def test_training_invalid_path():
    with pytest.raises(ValidationError):
        TrainRequest(data_path='invalid_path', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5)

def test_training_invalid_eval_matrix():
    with pytest.raises(ValidationError):
        TrainRequest(data_path='valid_path.csv', target_name='label', eval_matrix=['acc'], num_cv_fold=5)

def test_training_invalid_cv():
    with pytest.raises(ValidationError):
        TrainRequest(data_path='valid_path.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold="4.")

def test_training_invalid_dtree_value():
    with pytest.raises(ValidationError):
        TrainRequest(data_path='valid_path.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(criterion='xd'))

def test_training_invalid_dtree_field():
    with pytest.raises(ValidationError):
        TrainRequest(data_path='valid_path.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=5, dtree_param=DecisionTreeParam(wrong_field='xd'))

# train response
def test_training_response_valid():
    assert TrainResponse(model_id='model_id', cv_scores=Scores(accuracy=0.9, precision_micro=0.7, recall_micro=0.85))

def test_training_empty_score():
    with pytest.raises(ValidationError):
        TrainResponse(model_id='model_id', cv_scores=Scores())

def test_training_invalid_score_value():
    with pytest.raises(ValidationError):
        TrainResponse(model_id='model_id', cv_scores=Scores(accuracy='acc'))

def test_training_invalid_score_field():
    with pytest.raises(ValidationError):
        TrainResponse(model_id='model_id', cv_scores=Scores(acc=0.9))

def test_training_empty_id():
    with pytest.raises(ValidationError):
        TrainResponse(cv_scores=Scores(accuracy=0.9))

# test response
def test_testing_response_valid():
    assert TestResponse(scores=Scores(accuracy=0.9), preds=[3])

def test_testing_preds_wrong_type():
    with pytest.raises(ValidationError):
        TestResponse(scores=Scores(accuracy=0.9), preds=['pred'])
