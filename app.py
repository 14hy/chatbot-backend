from flask import Flask
from flask_cors import CORS
import config
from api import api

CONFIG = config.FLASK

if __name__ == '__main__':
    app = Flask(__name__)
    CORS(app)

    api.init_app(app=app)
    app.run(host=CONFIG['host'], port=CONFIG['port'], debug=CONFIG['debug'])
