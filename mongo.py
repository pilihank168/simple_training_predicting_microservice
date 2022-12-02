import pymongo
import datetime
import gridfs
from bson.objectid import ObjectId

def put_file(file):
    db = mongo_client.hello_db
    files = gridfs.GridFS(db)
    f_id = files.put(file)
    return f_id

def get_file(f_id):
    db = mongo_client.hello_db
    files = gridfs.GridFS(db)
    return files.get(f_id)

def insert_model(score, features, model, target_name, eval_matrix, cv_fold, dtree_param):
    f_id = put_file(model)
    models = mongo_client.hello_db.models
    new_entry = {'score':score, 'features':features, 'date_time':datetime.datetime.now(), 'model':f_id,
                    'target_name': target_name, 'eval_matrix': eval_matrix, 'cv_fold': cv_fold, 'dtree_param': dtree_param}
    model_id = models.insert_one(new_entry).inserted_id
    model_id_str = str(model_id)
    return model_id_str

def query_model(model_id_str):
    models = mongo_client.hello_db.models
    doc = models.find_one({'_id': ObjectId(model_id_str)})
    f_id = doc["model"]
    return get_file(f_id).read(), doc["features"], doc['target_name'], doc['eval_matrix']

def list_all():
    models = mongo_client.hello_db.models
    docs = []
    cursor = models.find({})
    docs = [{'cv_score':doc['score'], 'features':doc['features'], 'date_time':doc['date_time'].isoformat(timespec='seconds'),
                'id':str(doc['_id']), 'target_name':doc['target_name'],'eval_matrix':doc['eval_matrix'], 
                'cv_fold':doc['cv_fold'], 'dtree_param':doc['dtree_param']} for doc in cursor]
    return docs

mongo_client = pymongo.MongoClient('localhost', 27017)
