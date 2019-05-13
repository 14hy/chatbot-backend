from flask import Flask
from flask_pymongo import PyMongo
from flask_restplus import Api
from flask_cors import CORS

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/chatbot"
CORS(app)
api = Api(app, version='0.2', title='Hanyang-Chatbot', description='회사발표 5월 18일 학교 발표 6월 5일 제출')
v1 = api.namespace('v1', description='version_1')
db = PyMongo(app)

