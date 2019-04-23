from api.common.settings import *

from api.resources.chatbot import CategorizeChat, Manager, answer
#
# api.add_resource(CategorizeChat, '/chatbot', '/chatbot/<string:chat>')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8800, debug=True)

