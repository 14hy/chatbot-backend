from flask import Flask
from flask_pymongo import PyMongo
# from flask_restful import Api
from flask_restplus import Api
from flask_cors import CORS
from werkzeug.contrib.fixers import ProxyFix

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
CORS(app)
api = Api(app, version=0.1, title='Hanyang-Chatbot')
v1 = api.namespace('v1', description='version_1')
db = PyMongo(app)

