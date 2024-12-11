from django.urls import path
from stories import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('create/', views.generate_story, name='create_story'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

