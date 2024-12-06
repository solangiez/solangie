import streamlit as st
import numpy as np
from scipy.io import wavfile
import plotly.graph_objects as go

# Configuración inicial
st.title("Procesamiento y Modulación de Audio")
st.markdown("**Sube un archivo WAV para procesarlo y realizar todo el proceso de modulación.**")

# Subir archivo
uploaded_file = st.file_uploader("Subir archivo WAV", type=["wav"])

if "current_plot" not in st.session_state:
    st.session_state.current_plot = 0  # Índice del gráfico actual

def next_plot():
    if st.session_state.current_plot < len(plot_functions) - 1:
        st.session_state.current_plot += 1

def prev_plot():
    if st.session_state.current_plot > 0:
        st.session_state.current_plot -= 1

if uploaded_file is not None:
    # Leer archivo
    samplerate, data = wavfile.read(uploaded_file)
    length = data.shape[0] / samplerate
    st.write(f"Duración del audio: {length:.2f} segundos")
    
    # Reproducir el audio original
    st.subheader("Audio Original")
    st.audio(uploaded_file, format="audio/wav")
    
    time = np.linspace(0., length, data.shape[0])

    if len(data.shape) == 2:  # Si es estéreo
        x_t = data[:, 0] / np.max(data[:, 0])
    else:
        x_t = data / np.max(data)
    

    fm= samplerate*10
    st.write("su frecuencia de muestreo es ")

    # Parámetros de la señal portadora
    A = 1
    st.write("El valor seleccionado es " + str(fm) + " HZ")

    f0 = st.slider(
    label="Solo modifiqué, esto sí es necesario. De que está modificando la frecuencia de portador teniendo en cuenta la de muestreo",
    min_value=30000,  # Valor mínimo
    max_value=3000000,  # Valor máximo
    value=300000,  # Valor inicial
    step=30000  # Incremento
    )

    A = 1
    w0 = 2 * np.pi * f0
    p_t = A * np.cos(w0 * time)

    w = np.linspace(-samplerate / 2, samplerate / 2, len(x_t))
    x_w = np.fft.fft(x_t)
    x_w_centrado = np.abs(np.fft.fftshift(x_w))
    x_w_normalizado = x_w_centrado / np.max(x_w_centrado)

    x_mod = x_t * p_t
    x_w_mod = np.fft.fft(x_mod)
    x_w_mod_centrado = np.abs(np.fft.fftshift(x_w_mod))
    x_w_mod_normalizado = x_w_mod_centrado / np.max(x_w_mod_centrado)

    x_dem = x_mod * p_t
    x_w_dem = np.fft.fft(x_dem)
    x_w_dem_centrado = np.abs(np.fft.fftshift(x_w_dem))
    x_w_dem_normalizado = x_w_dem_centrado / np.max(x_w_dem_centrado)

    limit1 = int(-samplerate / 2 + len(w) / 2)
    limit2 = int(samplerate / 2 + len(w) / 2)
    x_w_recortado = x_w_dem_normalizado[limit1:limit2]

    x_dem_normalizada = x_dem / np.max(np.abs(x_dem))  # Normalizar entre -1 y 1
    x_dem_int16 = (x_dem * 32767).astype(np.int16)  # Convertir a formato int16


    

    # Funciones de graficado interactivo
    def plot_signal(time, signal, title, xlabel, ylabel, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time[:2000], y=signal[:2000], mode='lines', line=dict(color=color)))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
        st.plotly_chart(fig)

    def plot_fft(frequencies, fft_values, title, xlabel, ylabel, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies, y=fft_values, mode='lines', line=dict(color=color)))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template="plotly_white")
        st.plotly_chart(fig)

    def plot_original_signal():
        plot_signal(time, x_t, "Señal Original en el Tiempo", "Tiempo (s)", "Amplitud", "blue")

    def plot_original_fft():
        plot_fft(w, x_w_normalizado, "FFT de la Señal Original", "Frecuencia (Hz)", "Amplitud Normalizada", "red")

    def plot_modulated_signal():
        plot_signal(time, x_mod, "Señal Modulada en el Tiempo", "Tiempo (s)", "Amplitud", "green")

    def plot_modulated_fft():
        plot_fft(w, x_w_mod_normalizado, "FFT de la Señal Modulada", "Frecuencia (Hz)", "Amplitud Normalizada", "orange")

    def plot_demodulated_signal():
        plot_signal(time, x_dem, "Señal Demodulada en el Tiempo", "Tiempo (s)", "Amplitud", "purple")

    def plot_demodulated_fft():
        plot_fft(w, x_w_dem_normalizado, "FFT de la Señal Demodulada", "Frecuencia (Hz)", "Amplitud Normalizada", "cyan")

    def plot_filtered_signal():
        plot_fft(w[limit1:limit2], x_w_recortado, "FFT de la Señal Demodulada Filtrada (Pasa Bajas)", "Frecuencia (Hz)", "Amplitud Normalizada", "magenta")

    plot_functions = [
        plot_original_signal,
        plot_original_fft,
        plot_modulated_signal,
        plot_modulated_fft,
        plot_demodulated_signal,
        plot_demodulated_fft,
        plot_filtered_signal,
    ]

    st.subheader(f"Gráfico {st.session_state.current_plot + 1} de {len(plot_functions)}")
    plot_functions[st.session_state.current_plot]()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Anterior"):
            prev_plot()
    with col3:
        if st.button("Siguiente"):
            next_plot()
        wavfile_name = "audio_demodulado.wav"
    wavfile.write(wavfile_name, samplerate, x_dem_int16 )

    # Botón para descargar el archivo
    with open(wavfile_name, "rb") as file:
        st.download_button(
            label="Descargar audio demodulado",
            data=file,
            file_name="audio_demodulado.wav",
            mime="audio/wav"
        )

    # Reproducir el archivo WAV demodulado en Streamlit
    st.subheader("Audio Demodulado")
    st.audio(wavfile_name, format="audio/wav")