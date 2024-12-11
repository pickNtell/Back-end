from django.urls import path
from stories import views

urlpatterns = [
    path('', views.home, name='home'),
    path('create/', views.generate_story, name='create_story'),
]

