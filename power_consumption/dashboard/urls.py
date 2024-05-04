
from django.urls import path,include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [

    path('login/', views.login_page, name='login_page'),
    path('register/', views.register, name='register'),
    path('dash/', views.dash, name='dash'),
    path('streamlit/', views.streamlit_view, name='streamlit'),
    path('resetpass/',views.resetpass,name='resetpass'),

]
