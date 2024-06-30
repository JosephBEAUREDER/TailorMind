from django.urls import path
from . import views
from .views import save_texte

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('save_texte/', save_texte, name='save_texte'),
    path('create_user_db/', views.create_user_db, name='create_user_db'),
    path('check_user_db/', views.check_user_db, name='check_user_db'),
    path('query_rag/', views.query_rag, name='query_rag'),

]