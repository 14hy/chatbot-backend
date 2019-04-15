from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
import json

from engine.main import Engine
from server.models import Member

# from server.api import MemberSerializer

# Create your views here.

engine = Engine()


@csrf_exempt
def test_REST(request):
    if request.method == 'GET':
        # server = Member.objects.all()
        query_params = request.query_params.get('chat')
        print(query_params)
        return
        # serializer = MemberSerializer(server, many=True)\
        return HttpResponse(json.dumps(engine.chat_to_answer(chat),
                                       ensure_ascii=False), content_type="application/json; charset=utf-8")

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        print(data)
        # serializer = MemberSerializer(data=data)
        return JsonResponse('bye', status=400)
