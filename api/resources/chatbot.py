from flask_restplus import Resource
from flask_restplus import reqparse, fields, inputs
from flask import Response
from api.common.settings import *
from src.db.questions import index as questions
from src.db.contexts import index as contexts
from src.main import Engine
from src.services.analysis import *

backend = Engine()


@v1.route('/chat')
class CategorizeChat(Resource):

    @v1.doc(params={'chat': 'A chat'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('chat', type=str, required=True, help='사용자가 입력한 질문 및 발화')
            args = parser.parse_args()
            _chat = args['chat']
            _answer = backend.chat_to_answer(_chat)
            return _answer
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/db/questions/add')
class Questions(Resource):

    @v1.doc('질문 추가', params={'text': '등록 할 질문', 'answer': '등록 할 답변(default=None)', 'category': '카테고리'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('text', type=str, help='등록 할 질문')
            parser.add_argument('answer', type=str, required=False, help='답변(default=None)')
            parser.add_argument('category', type=str, required=True, help='카테고리')
            args = parser.parse_args(strict=True)

            _text = args['text']
            _answer = args['answer']
            _category = args['category']
            questions.create_insert(_text, _answer, _category)
            return {'status': 'Success'}
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/db/contexts/add')
class Contexts(Resource):

    @v1.doc('문단 추가', params={'subject': '등록 할 문단의 주제', 'text': '등록 할 문단'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('subject', type=str, help='등록 할 문단의 주제')
            parser.add_argument('text', type=str, required=True, help='등록 할 문단')
            args = parser.parse_args(strict=True)

            _subject = args['subject']
            _text = args['text']
            contexts.create_insert(text=_text, subject=_subject)
            return {'status': 'Success'}
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/bus/shuttle')
class Shuttle(Resource):

    @v1.doc('셔틀 버스 정보 조회', params={'weekend': '휴일여부(True, False)', 'season': 'semester/ between/ vacation',
                                   'hours': '시간 - int(0~23)', 'minutes': '분 - int(0~59)', 'seconds': '초 - int(0~59)'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('weekend', required=True, type=inputs.boolean, help='휴일여부')
            parser.add_argument('season', required=True, type=str, help='학기/ 계절학기/ 방학')
            parser.add_argument('hours', type=int, required=True, help='시간(0~23)')
            parser.add_argument('minutes', type=int, required=True, help='분(0~59)')
            parser.add_argument('seconds', type=int, required=True, help='초(0~59)')
            args = parser.parse_args(strict=True)

            _weekend = args['weekend']
            _season = args['season']
            _hours = args['hours']
            _minutes = args['minutes']
            _seconds = args['seconds']

            return backend.get_shuttle(_weekend, _season, _hours, _minutes, _seconds)
        except Exception as err:
            print(err)
            return {'error': str(err)}

    @v1.doc('셔틀 버스 정보 조회(현재시간)')
    def get(self):
        try:
            return backend.get_shuttle(current=True)
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/analysis/similarity/morphs')
class AnalysisSimilarityMorphs(Resource):

    @v1.doc('사용자 질문의 문장 형태소 분석 조회', params={'chat': '사용자 질문'})
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('chat', required=True, type=str, help='사용자 질문')
            args = parser.parse_args(strict=True)

            _chat = args['chat']
            return get_Morphs(_chat)
        except Exception as err:
            print(err)
            return {'error': str(err)}

    @v1.doc('사전 답변과의 문장 형태소 분석 조회', params={'chat': '사용자 질문'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('chat', required=True, type=str, help='사용자 질문')
            args = parser.parse_args(strict=True)

            _chat = args['chat']
            return get_JaccardSimilarity(_chat)
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/analysis/statistics/keywords')
class AnalysisStatisticsKeywords(Resource):

    @v1.doc('키워드 통계 조회', params={'number': '조회 할 개수'})
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('number', required=False, default=7, type=int, help='조회 할 개수(기본 7개)')
            args = parser.parse_args(strict=True)

            _number = args['number']
            return get_MostCommonKeywords(_number)
        except Exception as err:
            print(err)
            return {'error': str(err)}


@v1.route('/analysis/queries/searchs')
class AnalysisStatisticsKeywords(Resource):

    @v1.doc('쿼리에서 search 카테고리 모두 검색', params={'number': '검색 할 개수(기본 20개)'})
    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('number', required=False, default=20, type=int, help='조회 할 개수(기본 20개)')
            args = parser.parse_args(strict=True)

            _number = args['number']
            return get_SearchToQuestion(_number)
        except Exception as err:
            print(err)
            return {'error': str(err)}

