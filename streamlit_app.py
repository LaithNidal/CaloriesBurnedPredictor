#Importing libraries 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import pickle


#Configuring page setup 
st.set_page_config(page_title="Calories Burned Prediction (via Regression)", page_icon="🔥", layout="centered", initial_sidebar_state="expanded")
st.markdown("<div style = 'background-color: #f0f0f5; padding: 10px;'><h1 justify-content: center; style='text-align: center; color: black;'>Calories Burned Prediction</h1></div>", unsafe_allow_html=True)
st.markdown("<h4 style = 'text-align: center; color: black;'>This app predicts the calories burned during a workout based on the input parameters of your workout.</h4>", unsafe_allow_html=True)

#Loading the dataset

df = pd.read_csv("calories_burn_clean.csv")

#Styling Streamlit Web App

col1, col2 = st.columns(2)

with col1:
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.write("  ")
  st.image("calories.jpeg", use_container_width = True)

with col2:

  gender = st.radio(label = 'Select your gender (0 for male 1 for female)', options = df['Gender'].unique(), index= None)

  age = st.number_input(label = 'Enter your age',placeholder="Enter your age",value=None,min_value=0,max_value=99,step=1)

  col3, col4 = st.columns(2)
  with col3:
    height = st.number_input(label = 'Enter your height in cm',placeholder="Enter your height",value=None,min_value=0,max_value=230,step=1)
  with col4:
    weight = st.number_input(label = 'Enter your weight in kg',placeholder="Enter your  weight in kg",value=None,min_value=0,max_value=170,step=1)
  col5, col6 = st.columns(2)
  with col5:
    heart_rate = st.number_input(label = 'Enter your average heart rate during workout',placeholder="Enter your average heart rate during workout",value=None,min_value=0,max_value=200,step=1)
  with col6:
    body_temp = st.number_input(label = 'Enter your average body temperature during your workout',placeholder="Enter your average body temperature during your workout",value=None,min_value=0.0,max_value=41.0,step=0.1)
  col7 = st.columns(1)[0]
  with col7: 
    duration = st.number_input(label = 'Enter the duration of your workout in minutes',placeholder="Enter workout duration",value=None,min_value=0,max_value=200,step=1)
  pred = st.button("Predict", use_container_width = True)


X = df.drop(['User_ID', 'Calories'], axis = 1)
y = df['Calories']

#Feature Scaling 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)





# Setting up model with best parameters
model = joblib.load(open('calories_burned_model.joblib', 'rb')) # read-binary


  #Creating DataFrame
input_df = pd.DataFrame({'Gender':[gender], 'Age':[age], 'Height':[height], 'Weight':[weight], 'Heart Rate':[heart_rate], 'Body Temperature':[body_temp], 'Duration':[duration]})

df1 = pd.DataFrame(input_df)


  #Defining the correct for Columns 

model_features = ['Gender', 'Age', 'Height','Weight', 'Heart Rate', 'Body Temperature', 'Duration']

for feature in model_features: 
  if feature not in df1.columns: 
    df1[feature] = 0

df1 = df1[model_features]
df

#Making Prediction by Trained ML Model 

if pred:

  if any([gender is None, age is None, height is None, weight is None, heart_rate is None, body_temp is None]): 
    st.error("Please select all inputs before pressing the predict button.", icon ="📝")
  else: 
    y_pred = model.predict(df1)
    #acc = accuracy_score(y_val, y_pred)
    st.write(f'Calories burned during your  = {y_pred}')
