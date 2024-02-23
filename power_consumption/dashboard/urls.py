
from django.urls import path,include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [

    path('login/',views.login,name='login'),
    path('dash/',views.dash,name='dash'),
    path('resetpass/',views.reset,name='reset'),
    path('resetsuccess/',views.resetsuccess,name='resetsuccess'),
]
