from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    #return HttpResponse("Hello, world. You're at the polls index.")
    return render(request, 'core/index.html', {'condition': True, 'mylist': [1, 2, 3, 4]})