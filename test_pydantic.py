import pytest
from pydantic_model import *

# valid data path
def test_csv_data_path():
    assert valid_data_path('xd.csv')

def test_txt_data_path():
    assert valid_data_path('xd.txt')

def test_invalid_data_path():
    with pytest.raises(TypeError):
        valid_data_path('xddd')

# decision tree
def test_dtree_valid():
    assert DecisionTreeParam(criterion='entropy', splitter='random')

def test_dtree_extra_field():
    with pytest.raises(ValidationError):
        DecisionTreeParam(xd='xd')

def test_dtree_extra_value():
    with pytest.raises(ValidationError):
        DecisionTreeParam(criterion='xd')

# scores
def test_scores_valid():
    assert Scores(accuracy=0.8, precision_micro=0.7, recall_micro=0.85)

def test_scores_empty():
    with pytest.raises(ValidationError):
        Scores()

def test_scores_extra_field():
    with pytest.raises(ValidationError):
        Scores(f1=3)

def test_scores_invalid_value():
    with pytest.raises(ValidationError):
        Scores(accuracy='acc')

# train request
def test_training_invalid_eval_matrix():
    with pytest.raises(ValidationError):
        assert TrainRequest(data_path='xd.csv', target_name='label', eval_matrix=['acc'], num_cv_fold=5)

def test_training_invalid_cv():
    with pytest.raises(ValidationError):
        assert TrainRequest(data_path='xd.csv', target_name='label', eval_matrix=['accuracy'], num_cv_fold=4.5)

# train response
def test_training_response_valid():
    assert TrainResponse(model_id='model_id', cv_scores=Scores(accuracy=0.9))

def test_training_empty():
    with pytest.raises(ValidationError):
        assert TrainResponse()

# test request
def test_testing_request_valid():
    assert TestRequest(data_path='xd.csv', model_id='model_id')

def test_testing_empty_id():
    with pytest.raises(ValidationError):
        TestRequest(data_path='xd.csv')

# test response
def test_testing_response_valid():
    assert TestResponse(scores=Scores(accuracy=0.9), preds=[3])

def test_testing_preds_wrong_type():
    with pytest.raises(ValidationError):
        TestResponse(scores=Scores(accuracy=0.9), preds=['pred'])
