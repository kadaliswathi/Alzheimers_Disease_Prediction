from django.shortcuts import render, HttpResponse
from django.contrib import messages

from Alzheimers_Disease_Prediction.settings import MEDIA_ROOT
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import default_storage
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ── Model Setup (loaded once at startup) ──────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'alzheimer_final_model.keras')
RESULT_PATH = os.path.join(settings.MEDIA_ROOT, 'training_results.json')
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)


# ── Auth Views ────────────────────────────────────────────────────────────────
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd    = request.POST.get('pswd')
        try:
            check  = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            if status == "activated":
                request.session['id']        = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid']   = loginid
                request.session['email']     = check.email
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account is not activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    import pandas as pd
    path    = os.path.join(settings.MEDIA_ROOT, 'alzheimers_prediction_dataset.csv')
    df      = pd.read_csv(path)
    df_html = df.to_html(classes='table table-bordered table-striped', index=False)
    return render(request, 'users/viewdataset.html', {'data': df_html})


# ── Training View ─────────────────────────────────────────────────────
    import os
import json
from django.conf import settings
from django.shortcuts import render

# ✅ Use MEDIA_ROOT instead of hardcoded path
RESULT_PATH = os.path.join(settings.MEDIA_ROOT, 'training_result.json')


def Training(request):
    context = {'trained': False, 'error': None}

    if os.path.exists(RESULT_PATH):
        try:
            with open(RESULT_PATH, 'r') as f:
                data = json.load(f)

            context.update(data)
            context['trained'] = True

        except Exception as e:
            context['error'] = f"Error loading JSON: {str(e)}"

    else:
        context['error'] = "Training results file not found."

    return render(request, 'users/training.html', context)


# ── Prediction View ───────────────────────────────────────────────────────────
import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ✅ Model Path
MODEL_PATH = os.path.join(MEDIA_ROOT, 'best_alzheimer_efficientnetb0.keras')
# ✅ Load model ONCE
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class labels (must match training order)
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']


import numpy as np
from PIL import Image

def is_grayscale(img_path):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)

    # Split channels
    R, G, B = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

    # Check if channels are almost equal
    if np.allclose(R, G, atol=10) and np.allclose(G, B, atol=10):
        return True
    return False


def Prediction(request):
    context = {'prediction': None}

    if request.method == 'POST' and request.FILES.get('mri_image'):

        if model is None:
            context['prediction'] = "Error: Model not found."
            return render(request, 'users/predict_form.html', context)

        img_file = request.FILES['mri_image']

        saved_path = default_storage.save(f'tmp/{img_file.name}', img_file)
        full_path = default_storage.path(saved_path)

        try:
            # ✅ CHECK grayscale first
            if not is_grayscale(full_path):
                context['prediction'] = "❌ Invalid Image! Please upload MRI."
                return render(request, 'users/predict_form.html', context)

            # ✅ Load image
            img = image.load_img(full_path, target_size=(224, 224))
            img_array = image.img_to_array(img)

            # ✅ Preprocess
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # ✅ Predict
            preds = model.predict(img_array)

            predicted_class = CLASS_NAMES[np.argmax(preds)]
            confidence = round(float(np.max(preds)) * 100, 2)

            context['prediction'] = predicted_class
            context['confidence'] = confidence

        except Exception as e:
            context['prediction'] = f"Error: {str(e)}"

        finally:
            default_storage.delete(saved_path)

    return render(request, 'users/predict_form.html', context)