import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath
#title
st.title("Sport to'plarini klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader('Rasmni yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)
    #model
    model = load_learner('sportball_model.pkl')

    #prediction
    pred, pred_id, probs=model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimolik: {probs[pred_id]*100:.1f} %')
    
    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)


