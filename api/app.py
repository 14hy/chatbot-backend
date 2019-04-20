from api.common.settings import *

from api.resources.chatbot import CategorizeChat, Manager, answer
#
# api.add_resource(CategorizeChat, '/chatbot', '/chatbot/<string:chat>')



if __name__ == '__main__':
    app.run(port=4001, debug=True)

