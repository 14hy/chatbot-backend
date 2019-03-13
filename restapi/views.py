import ast

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

from engine.main import Engine
from restapi.models import Member
# from restapi.api import MemberSerializer

# Create your views here.

engine = Engine()

@csrf_exempt
def test_REST(request):
    """
    List all code members, or create a new snippet.
    """
    if request.method == 'GET':
        # restapi = Member.objects.all()
        question = request.GET['question']
        question = ast.literal_eval(question)
        print(question)
        # serializer = MemberSerializer(restapi, many=True)
        return JsonResponse(engine.question_to_answer(question[0]),
                            safe=False)
    #
    # elif request.method == 'POST':
    #     data = JSONParser().parse(request)
    #     # serializer = MemberSerializer(data=data)
    #     if serializer.is_valid():
    #         serializer.save()
    #         return JsonResponse(serializer.data, status=201)
    #     return JsonResponse('bye', status=400)
