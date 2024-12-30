import joblib
from django.shortcuts import render
from django.http import JsonResponse
# predictor/views.py

from django.shortcuts import render

def home(request):
    return render(request, 'predictor/index.html')


# Load the trained model
model = joblib.load('space_weather_model.pkl')

def index(request):
    return render(request, 'predictor/index.html')

def predict(request):
    # Get input from the frontend (e.g., sunspot count, solar cycle)
    sunspot_count = float(request.GET.get('sunspot_count'))
    solar_cycle = int(request.GET.get('solar_cycle'))

    # Make prediction
    prediction = model.predict([[sunspot_count, solar_cycle]])[0]

    return JsonResponse({'prediction': prediction})

# Create your views here.
