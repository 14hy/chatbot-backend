from flask import Flask
from flask_pymongo import PyMongo
# from flask_restful import Api
from flask_restplus import Api


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
api = Api(app)
v1 = api.namespace('v1', description='version_1')
db = PyMongo(app)

