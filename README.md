# CaloriesBurnedPredictor

# Calories Burned Prediction Web App

This project aims to predict calories burned during a workout using machine learning in the form of a supervised regression model. The core of the project involves preprocessing data to build a regression model to estimate calorie expenditure based on several user-specific features. The workflow includes data loading, preprocessing, model training, and deployment as a web application.

The solution uses a RandomForestRegressor model, trained on preprocessed data. Key steps in the data handling phase included feature scaling to ensure optimal model performance. The trained model is saved and then integrated into a Streamlit application.

The Streamlit application provides a user-friendly interface where users can input their workout parameters, such as gender, age, height, weight, heart rate,duration and body temperature. The application then uses the trained machine learning model to predict the number of calories burned and displays the result. You can try the application using the following link: https://caloriesburnedpredictor1.streamlit.app/

