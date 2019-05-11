from pymongo import MongoClient

import config

MONGODB_CONFIG = config.MONGODB

client = MongoClient(host=MONGODB_CONFIG['local_ip'],
                     port=MONGODB_CONFIG['port'],
                     username=MONGODB_CONFIG['username'],
                     password=MONGODB_CONFIG['password'])

db = client[MONGODB_CONFIG['db_name']]
