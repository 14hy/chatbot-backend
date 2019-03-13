from django.urls import path
from restapi import views

urlpatterns = [
    path('api/', views.test_REST),
    # path('snippets/<int:pk>/', views.snippet_detail),
]
