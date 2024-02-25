import pickle
import numpy as np
from sklearn.linear_model import Perceptron
import streamlit as st

model = pickle.load(open('per_model-65130700329.sav', 'rb'))

st.title("Iris Specie Prediction using Perceptron")

x1 = st.slider('Select Input1', 0.0 , 10.0 , 3.0)
x2 = st.slider('Select Input1', 0.0 , 10.0 , 5.0)
x3 = st.slider('Select Input1', 0.0 , 10.0 , 4.0)
x4 = st.slider('Select Input1', 0.0 , 10.0 , 7.0)

xnew = np.array([[x1,x2,x3,x4]])


pred = model.predict(xnew)

st.write("## prediction Result:")
st.write("Specites:", pred[0])
