from flask import Flask
from flask_pymongo import PyMongo
# from flask_restful import Api
from flask_restplus import Api
from werkzeug.contrib.fixers import ProxyFix


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
api = Api(app)
ns = api.namespace('test', description='test')
db = PyMongo(app)

