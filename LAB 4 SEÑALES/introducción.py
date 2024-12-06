import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
   page_title= "multipage"

)
st.title("Interfaz gráfica de Fourier")
st.sidebar.success("Selecciona algún proceso")
video_url = "https://www.youtube.com/watch?v=Mdk6BWeVNIs&t=198s&ab_channel=MatesMike"  # Reemplaza con tu enlace
st.video(video_url)
st.markdown("<h1 style='font-size:40px;'>", unsafe_allow_html=True)

video_url2 = "https://www.youtube.com/watch?v=rwN6UkRF9bY&ab_channel=PlataformaEducativaArag%C3%B3n-IEE"  # Reemplaza con tu enlace
st.video(video_url2)
st.markdown("<h1 style='font-size:40px;'>", unsafe_allow_html=True)