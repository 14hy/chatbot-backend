from flask import Flask
from flask_pymongo import PyMongo
from flask_restplus import Api
from flask_cors import CORS

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
CORS(app)
api = Api(app, version=0.2, title='Hanyang-Chatbot', description='수장이형은 우리가 이렇게 고생하는 걸 알까?')
v1 = api.namespace('v1', description='version_1')
db = PyMongo(app)

