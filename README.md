# Alzheimer's Disease Prediction using Modern Machine Learning Techniques

An end-to-end intelligent healthcare web application built with **Django** and **Deep Learning (EfficientNet-B0)** to detect and classify Alzheimer's Disease stages from brain MRI scans.

---

## 📌 Project Overview

Alzheimer's disease is an irreversible, progressive brain disorder that slowly destroys memory and cognitive skills. Early and accurate detection plays a vital role in providing timely care and treatment planning.

This project implements a web-based diagnosis assistance system where medical practitioners and users can upload brain MRI scans. A fine-tuned **EfficientNet-B0** deep convolutional neural network analyzes the scan, validates that it is a valid grayscale MRI image, and predicts the specific disease stage with high confidence.

### 🧠 Classification Stages:
1. **Mild Demented**
2. **Moderate Demented**
3. **Non-Demented (Healthy)**
4. **Very Mild Demented**

---

## ✨ Features

- **User Portal**:
  - Secure User Registration and Authentication.
  - Interactive MRI Image Upload form.
  - Real-time MRI validation (grayscale verification check).
  - Instant classification prediction with confidence score.
  - Interactive Dataset Explorer to view clinical metrics and metadata.
  - Training metrics dashboard (accuracy, loss curves, confusion matrix, F1-scores).

- **Admin Management**:
  - Admin login & dashboard (/admin and /AdminLogin).
  - User verification and account activation control.
  - Overall system oversight.

- **Deep Learning Pipeline**:
  - Pretrained **EfficientNet-B0** architecture fine-tuned on Alzheimer MRI data.
  - Preprocessed with standard EfficientNet input scaling (224x224).
  - Model weights and architecture stored in media/best_alzheimer_efficientnetb0.keras and media/model.weights.h5.

- **Ready-to-Test Sample Scans**:
  - Includes sample MRI brain scans inside sample_test_mri_images/ covering all 4 disease stages for immediate testing and demonstration.

---

## 🛠️ Tech Stack

- **Backend Framework**: Django 5.1 (Python)
- **Deep Learning / AI**: TensorFlow 2.x, Keras, EfficientNet-B0
- **Data Analysis**: NumPy, Pandas, Pillow (PIL), Scikit-Learn
- **Frontend**: HTML5, CSS3, Bootstrap 4, JavaScript, FontAwesome
- **Database**: SQLite (default Django ORM)

---

## 📁 Directory Structure

`	ext
Alzheimers_Disease_Prediction/
├── admins/                          # Admin views, user activation, administrative routes
├── users/                           # User views, model inference, forms, authentication
│   ├── views.py                     # MRI validation and prediction inference logic
│   ├── models.py                    # User registration database schema
│   └── forms.py                     # Registration form validation
├── Alzheimers_Disease_Prediction/   # Django project configuration (settings, URLs, WSGI/ASGI)
├── templates/                       # Frontend HTML templates
│   ├── admins/                      # Admin dashboard templates
│   ├── users/                       # Prediction form, training metrics, dataset views
│   └── base.html                    # Base template layout
├── static/                          # CSS stylesheets, JavaScript files, fonts, and images
├── media/                           # Deep learning models, training charts, dataset CSV
│   ├── best_alzheimer_efficientnetb0.keras  # Saved model file
│   ├── model.weights.h5             # Model weights file
│   ├── acc.png, cm.png, f1.png, loss.png  # Performance evaluation charts
│   ├── alzheimers_prediction_dataset.csv
│   └── balanced_data.csv
├── sample_test_mri_images/          # Sample MRI brain scans for quick testing
│   ├── MildDemented/                # Sample Mild Demented scans
│   ├── ModerateDemented/            # Sample Moderate Demented scans
│   ├── NonDemented/                 # Sample Non-Demented (Healthy) scans
│   └── VeryMildDemented/            # Sample Very Mild Demented scans
├── manage.py                        # Django management command-line utility
├── .gitignore                       # Git exclusion rules
└── README.md                        # Documentation
`

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Git installed.

### 2. Clone the Repository
`ash
git clone https://github.com/kadaliswathi/Alzheimers_Disease_Prediction.git
cd Alzheimers_Disease_Prediction
`

### 3. Create and Activate Virtual Environment
`ash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
`

### 4. Install Dependencies
`ash
pip install django tensorflow pandas numpy pillow scikit-learn
`

### 5. Apply Database Migrations
`ash
python manage.py makemigrations
python manage.py migrate
`

### 6. Create Superuser (Optional for Admin Portal)
`ash
python manage.py createsuperuser
`

### 7. Run Development Server
`ash
python manage.py runserver
`

Open your browser and navigate to:
`
http://127.0.0.1:8000/
`

---

## 🧪 Testing Predictions with Sample Scans

To test the application:
1. Log in to the User Portal.
2. Navigate to the **Prediction** page (/prediction/).
3. Click **Choose File** and select any sample MRI brain scan from the sample_test_mri_images/ directory:
   - sample_test_mri_images/MildDemented/
   - sample_test_mri_images/ModerateDemented/
   - sample_test_mri_images/NonDemented/
   - sample_test_mri_images/VeryMildDemented/
4. Click **Predict**. The system will validate the scan and display the diagnosed stage with confidence percentage.

---

## 📊 Model Evaluation & Metrics

The deep learning model achieves high classification accuracy across all 4 stages:
- **Accuracy Curve**: media/acc.png
- **Loss Curve**: media/loss.png
- **Confusion Matrix**: media/cm.png
- **F1 Score Graph**: media/f1.png

---

## 🔒 Security & Privacy

- Secret keys, local development database (db.sqlite3), virtual environments, and temporary prediction files are ignored from source control via .gitignore.
- Production deployments should set DEBUG = False and configure a production-ready secret key via environment variables.

---

## 👩‍💻 Author

- **GitHub**: [@kadaliswathi](https://github.com/kadaliswathi)