import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Header and note
st.write('''
# A weight loss-free risk calculator for malnutrition in patients with cancer
*Note*: This app predicts the Global Leadership Initiative on Malnutrition (GLIM)-defined malnutrition in those patients 
***with normal body mass index and normal muscle mass*** (both as per the GLIM's
phenotypic criteria) but ***without weight loss information***.
*Version 1.0.0 by Liangyu Yin liangyuyin1988@qq.com*
        ''')

st.sidebar.header('User Input Parameters')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/kevinlyy/data/GLIM_example.csv)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        food_intake_now = st.sidebar.selectbox('Food intake now',('Normal', 'Slightly reduced (less than 25%)', 'Severely reduced (25% or higher)'))
        anorexia = st.sidebar.selectbox('Anorexia',('yes','no'))
        diarrhea = st.sidebar.selectbox('Diarrhea',('yes','no'))
        taste_changes = st.sidebar.selectbox('Taste changes',('yes','no'))
        early_satiety = st.sidebar.selectbox('Early satiety',('yes','no'))
        pain = st.sidebar.selectbox('Pain',('yes','no'))
        ECOG_reduced = st.sidebar.selectbox('Reduced physical performance',('yes','no'))
        prealbumin = st.sidebar.slider('Prealbumin, mg/L', 1,1000,100) # min, max, default
        hemoglobin = st.sidebar.slider('Hemoglobin, g/L', 1,300,50)
        calf_circumference = st.sidebar.slider('Calf circumference, cm', 5.0, 80.0, 30.0)
        ASMI = st.sidebar.slider('Appendicular skeletal muscle index, kg/m^2', 3.0,15.0,7.0)
        data = {'food_intake_now':food_intake_now,
            'anorexia':anorexia,
            'diarrhea':diarrhea,
            'taste_changes':taste_changes,
            'early_satiety':early_satiety,
            'pain':pain,
            'ECOG_reduced':ECOG_reduced,
            'prealbumin':prealbumin,
            'hemoglobin':hemoglobin,
            'calf_circumference':calf_circumference,
            'ASMI':ASMI
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

GLIM_raw = pd.read_csv('GLIM_example.csv')
GLIM_data = GLIM_raw.drop(columns=['GLIM'])
df = pd.concat([input_df,GLIM_data],axis=0)

encode = ['food_intake_now','anorexia','diarrhea','taste_changes','early_satiety','pain','ECOG_reduced']

for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)

st.subheader('User Input Parameters')

if  uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)
    
load_model = pickle.load(open('GLIM_gbdt_app.pkl','rb'))

prediction = load_model.predict(df) # predict the class
prediction_proba = load_model.predict_proba(df) # predict the probability

st.subheader('Class labels and their corresponding index number')
target_names = {0:'Well-nourished',
                1:'Malnourished'}
target = pd.DataFrame(target_names, index=[0])
st.write(target)

st.subheader('Prediction')
st.write(target[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)