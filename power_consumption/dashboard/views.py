from django.shortcuts import render
from django.http import HttpResponse,request

# Create your views here.

def index(request):
    return HttpResponse("Index Dashboard")

def login(request):
    return render(request, 'dashboard/loginpage.html')

def dash(request):
    return render(request, 'dashboard/dashdisplay.html')
def reset(request):
    return render(request, 'dashboard/resetpass.html')
def resetsuccess(request):
    return render(request, 'dashboard/resetsuccess.html')


