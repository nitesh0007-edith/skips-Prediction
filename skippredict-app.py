import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

st.write("""
# Spotify Skip Prediction App

This app predict whether the song will be skipped or not

#### Author: Nitesh Ranjan Singh
""")

st.header('User input features')

load_clf = pickle.load(open('skip_predictionxgbc.pkl','rb'))

def classify(num):
    if num == 1:
        return 'Skipped'
    else:
        return 'Not-Skipped'


def user_input_features():
    session_position = st.slider('Session position',1,20)
    session_length = st.slider('Session position',10,20)
    context_switch = st.selectbox('Context switch', (0, 1))
    no_pause_before_play = st.slider('No pause before play', 0,1)
    short_pause_before_play = st.slider('short pause before play', 0,1)
    long_pause_before_play = st.slider('Long pause before play', 0,1)
    hour_of_day = st.slider('Hour of day', 0,23)
    premium = st.selectbox('Premium',('True','False'))
    context_type = st.selectbox('Context type',('editorial_playlist', 'user_collection', 'charts', 'catalog','radio', 'personalized_playlist'))
    hist_user_behavior_reason_start = st.selectbox('User behavior reason start',('trackdone', 'fwdbtn', 'backbtn', 'clickrow', 'appload', 'playbtn','remote', 'endplay', 'trackerror'))
    hist_user_behavior_reason_end = st.selectbox('User behavior reason end',('trackdone','fwdbtn','backbtn','endplay','remote','logout','clickrow'))
    duration = st.slider('Duration', 30.0, 1800.0)
    release_year = st.slider('Release_year', 1950, 2019)
    us_popularity_estimate = st.slider('US popularity estimate', 90.0, 100.0)
    beat_strength = st.slider('Beat strength', 0.0, 1.0)
    danceability = st.slider('Dance ability', 0.0, 1.0)
    instrumentalness = st.slider('Instrumentalness', 0.0, 1.0)
    liveness = st.slider('Liveness', 0.0, 1.0)
    loudness = st.slider('Loudness', -25.0, 0.0)
    mechanism = st.slider('Mechanism', 0.0, 1.0)
    mode = st.selectbox('Mode',('minor', 'major'))
    organism = st.slider('Organism', 0.0, 1.0)
    

    data = {'session_position': session_position,
    'session_length': session_length,
    'context_switch': context_switch,
    'no_pause_before_play': no_pause_before_play,
    'short_pause_before_play': short_pause_before_play,
    'long_pause_before_play': long_pause_before_play,
    'hour_of_day': hour_of_day,
    'premium': premium,
    'context_type': context_type,
    'hist_user_behavior_reason_start': hist_user_behavior_reason_start,
    'hist_user_behavior_reason_end': hist_user_behavior_reason_end,
    'duration': duration,
    'release_year': release_year,
    'us_popularity_estimate': us_popularity_estimate,
    'acousticness': acousticness,
    'beat_strength': beat_strength,
    'danceability': danceability,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'loudness': loudness,
    'mechanism': mechanism,
    'mode': mode,
    'organism': organism,
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()


objList = input_df.select_dtypes(include = "object").columns

for feat in objList:
    input_df[feat] = le.fit_transform(input_df[feat].astype(str))

df = input_df[:1]

if st.button('Classify'):
         st.subheader('User Input features')
         st.write(df)
         prediction = load_clf.predict(df)
         st.subheader('Prediction')
         skip = np.array(['Not Skipped','Skipped'])
         st.write(skip[prediction])
