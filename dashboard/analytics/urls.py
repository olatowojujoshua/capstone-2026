from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('overview/', views.overview, name='overview'),
    path('eda/', views.eda, name='eda'),
    path('models/', views.model_comparison, name='models'),
    path('volatility/', views.volatility, name='volatility'),
    path('fairness/', views.fairness, name='fairness'),
]