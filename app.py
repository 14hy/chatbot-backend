from api.common.settings import *
from api.resources.chatbot import CategorizeChat, Questions, Contexts, Shuttle

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
