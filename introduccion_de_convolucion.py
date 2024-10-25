import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
   page_title= "multipage"

)
st.title("interfaz grafica de Convolucion")
st.sidebar.success("seletc a page above")
video_url = "https://www.youtube.com/watch?v=BsmJDt96vz4&ab_channel=Dr.XavierFING"  # Reemplaza con tu enlace
st.video(video_url)
st.markdown("<h1 style='font-size:40px;'>Creado por David Serrano", unsafe_allow_html=True)
