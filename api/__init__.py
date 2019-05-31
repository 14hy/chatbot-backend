from flask_restplus import Api
from .v1.chatbot import v1 as ns_v1
from .v2.database.answer import api as ns_v2_db_question
from .v2.database.query import api as ns_v2_db_query
from .v2.database.context import api as ns_v2_db_context
from .v2.database.voca import api as ns_v2_db_voca
from .v2.preprocess.clean import api as ns_v2_prep_clean
from .v2.preprocess.keyword import api as ns_v2_prep_keyword
from .v2.preprocess.tag import api as ns_v2_prep_tag
from .v2.preprocess.token import api as ns_v2_prep_token
from .v2.service.bus import api as ns_v2_service_bus
from .v2.service.QA import api as ns_v2_service_QA
from .v2.visualize.bar import api as ns_v2_visualize_bar
from .v2.visualize.doughnut import api as ns_v2_visualize_doughnut
from .v2.visualize.histogram import api as ns_v2_visualize_histogram
from .v2.visualize.line import api as ns_v2_visualize_line
from .v2.visualize.scatter import api as ns_v2_visualize_scatter
from .v2.visualize.diagram import api as ns_v2_visualize_diagram
from .v2.visualize.table import api as ns_v2_visualize_table
from .v2.index import api as ns_v2
import config

CONFIG = config.FLASK

api = Api(version=CONFIG['version'], title=CONFIG['title'], description=CONFIG['desc'])

# version-1
api.add_namespace(ns_v1)

# version-2
api.add_namespace(ns_v2)
api.add_namespace(ns_v2_db_question)
api.add_namespace(ns_v2_db_query)
api.add_namespace(ns_v2_db_context)
api.add_namespace(ns_v2_db_voca)
api.add_namespace(ns_v2_prep_clean)
api.add_namespace(ns_v2_prep_keyword)
api.add_namespace(ns_v2_prep_tag)
api.add_namespace(ns_v2_prep_token)
api.add_namespace(ns_v2_service_bus)
api.add_namespace(ns_v2_service_QA)
api.add_namespace(ns_v2_visualize_bar)
api.add_namespace(ns_v2_visualize_doughnut)
api.add_namespace(ns_v2_visualize_histogram)
api.add_namespace(ns_v2_visualize_table)
api.add_namespace(ns_v2_visualize_line)
api.add_namespace(ns_v2_visualize_diagram)
api.add_namespace(ns_v2_visualize_scatter)
