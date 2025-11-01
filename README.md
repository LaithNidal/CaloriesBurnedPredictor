# 🏋️‍♂️Calories Burned Prediction:Supervised ML Regression Model (Random Forest Regressor) - Hosted on Streamlit Web App

**🚀 Project Overview**

The goal is to estimate calorie expenditure based on user-specific workout and physiological data.
The model leverages RandomForestRegressor to deliver accurate, real-time predictions of calories burned utilizing a supervised regression model

The core of the project involves preprocessing data to build a regression model to estimate calorie expenditure based on several user-specific features. The workflow includes data loading, preprocessing, model training, and deployment as a web application.

**📊 Workflow**

1. Data Loading & Cleaning – Imported and inspected raw data to ensure quality.
2. Feature Engineering – Processed relevant features such as gender, age, height, weight, heart rate, duration, and body temperature.
3. Feature Scaling – Normalized numeric columns for better model performance.
4. Model Training – Trained a **RandomForestRegressor** on the preprocessed training dataset.
5. Model Deployment – Saved the trained model using `pickle` and integrated it into a Streamlit web app.


**🧠 Model**

The Random Forest Regressor was chosen for its ability to handle non-linear relationships and reduce overfitting through ensemble learning.
Model evaluation included metrics such as **R² score 0f 0.997** (after hyperparameter tuning), and **Root Mean Squared Error (RMSE) value of 2.97** to validate performance.


**🌐 Streamlit App**

The web app provides a simple, interactive interface where users can input their personal details and workout data to instantly get a calorie prediction.

🔗 Try it here: Calories Burned Predictor

Inputs include:

* Genderv
* Age
* Height
* Weight
* Heart Rate
* Duration
* Body Temperature

**Output: Estimated Calories Burned 🔥**

**🧩 Tech Stack**

* Python
* Pandas, NumPy, Scikit-learn
* Streamlit
* Pickle

**📁 Repository Contents**

Assingment4-SupervisedML-Regression.ipynb – Jupyter notebook for data exploration, model training, and evaluation

model.pkl – Trained Random Forest model

app.py – Streamlit web application
