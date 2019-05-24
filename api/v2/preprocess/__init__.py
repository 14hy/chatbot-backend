from bson.objectid import ObjectId
from flask_restplus import Resource, Namespace
from flask_restplus import reqparse

api = Namespace('v2/preprocess', description='Proprocessor endpoint')