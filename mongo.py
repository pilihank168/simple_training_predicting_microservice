import pymongo
import datetime
import gridfs
from bson.objectid import ObjectId

def put_file(client, file):
    db = client.hello_db
    files = gridfs.GridFS(db)
    f_id = files.put(file)
    return f_id

def get_file(client, f_id):
    db = client.hello_db
    files = gridfs.GridFS(db)
    return files.get(f_id)

def insert_model(score=0.8, features=[0, 3], model=b'Hello mongodb and gridfs'):
    f_id = put_file(client, model)
    models = client.hello_db.models
    print(datetime.datetime.now())
    model_id = models.insert_one({'score':score, 'features':features, 'date_time':datetime.datetime.now(), 'model':f_id}).inserted_id
    model_id_str = str(model_id)
    print(model_id_str)
    return model_id_str

def query_model(model_id_str):
    models = client.hello_db.models
    doc = models.find_one({'_id': ObjectId(model_id_str)})
    f_id = doc["model"]
#    print(get_file(client, f_id).read())
    print(datetime.datetime.now())
    print(f'score: {doc["score"]}, features: {doc["features"]}, data_time: {doc["date_time"]}')
    return get_file(client, f_id).read(), doc["features"]

def list_all():
    models = client.hello_db.models
    docs = []
    cursor = models.find({})
    for doc in cursor:
        print(doc)
        docs.append({'cv_score':doc['score'], 'features':doc['features'], 'date_time':doc['date_time'].isoformat(timespec='seconds'), 'id':str(doc['_id'])})
    return docs

client = pymongo.MongoClient('localhost', 27017)
#list_all(client)
#m_id = insert_model(client)
#query_model(client, m_id)
