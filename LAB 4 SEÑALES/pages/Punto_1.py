import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

st.title("Gráfica de coeficientes de Fourier y sus representaciones en el espectro")
st.write(" Las siguientes señales a reconstruir son las siguientes")
st.latex(r'''
x(t) =
\begin{cases} 
    1 + 4 \frac{t}{T}, & \text{si } -\frac{T}{2} < t \leq 0, \\
    1 - 4 \frac{t}{T}, & \text{si } 0 \leq t < \frac{T}{2}, 
\end{cases} \tag{a}
''')
st.latex( r'''
x(t) = t \quad \text{si } -\pi \leq t \leq \pi
\tag{b}
''')
st.latex( r'''
x(t) = t^2 \quad \text{si } -\pi \leq t \leq \pi
\tag{c}
''')
st.latex(r'''
x(t) =
\begin{cases} 
    t, & \text{si } -1 < t < 0 \\
    1, & \text{si } 0 < t < 1 , 
\end{cases} \tag{d}
''')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Definición de las señales periódicas
def signal_a(t, T=8):
    """Triangular"""
    t = t % T
    if 0 <= t < T / 2:
        return 1 - (4 * t / T)
    elif T / 2 <= t < T:
        return -1 + (4 * (t - T/2) / T)
    return 0

def signal_b(t, T=10):
    """Abs(Seno)"""
    t = t % T
    return np.abs(np.sin(2 * np.pi * t / T))

def signal_c(t, T=10):
    """Parabólica"""
    t = t % T - T/2
    return t**2

def signal_d(t, T=2):
    """Lineal segmentada"""
    t = t % T
    if 0 <= t < T / 2:
        return 2 * t / T
    elif T / 2 <= t < T:
        return 2 - (2 * t / T)
    return 0

# Diccionario para mapear las señales a sus definiciones
signals = {
    "Señal A": signal_a,
    "Señal B": signal_b,
    "Señal C": signal_c,
    "Señal D": signal_d
}

# Cálculo de los coeficientes de Fourier
def calculate_fourier_coefficients(signal, T, N=10):
    """Calcula los coeficientes de Fourier hasta N armónicos."""
    a0 = (2 / T) * quad(lambda t: signal(t, T), 0, T)[0]
    an = []
    bn = []

    for n in range(1, N + 1):
        an.append((2 / T) * quad(lambda t: signal(t, T) * np.cos(2 * np.pi * n * t / T), 0, T)[0])
        bn.append((2 / T) * quad(lambda t: signal(t, T) * np.sin(2 * np.pi * n * t / T), 0, T)[0])

    return a0, an, bn

# Reconstrucción de la señal
def reconstruct_signal(a0, an, bn, T, t):
    """Reconstruye la señal a partir de sus coeficientes de Fourier."""
    reconstruction = a0 / 2
    for n in range(1, len(an) + 1):
        reconstruction += an[n-1] * np.cos(2 * np.pi * n * t / T) + bn[n-1] * np.sin(2 * np.pi * n * t / T)
    return reconstruction

# Interfaz de usuario en Streamlit
st.title("Análisis de Señales con Series de Fourier")
st.markdown("Selecciona una señal para visualizar su espectro y reconstrucción:")

# Selector para elegir la señal
signal_name = st.selectbox("Selecciona una señal", list(signals.keys()))

# Número de armónicos
N = st.slider("Número de armónicos (N)", min_value=1, max_value=50, value=10)

# Parámetros de la señal
t = np.linspace(-10, 10, 1000)  # Intervalo de tiempo
signal_function = signals[signal_name]
T = 8 if signal_name == "Señal A" else (10 if signal_name in ["Señal B", "Señal C"] else 2)

# Cálculo de coeficientes y reconstrucción
a0, an, bn = calculate_fourier_coefficients(signal_function, T, N)
reconstructed_signal = reconstruct_signal(a0, an, bn, T, t)

# Gráficos
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Espectro en línea
n_values = np.arange(0, N + 1)
coefficients = [a0 / 2] + [np.sqrt(a**2 + b**2) for a, b in zip(an, bn)]

ax[0].stem(n_values, coefficients, basefmt="r-")
ax[0].set_title("Espectro en Línea")
ax[0].set_xlabel("nω0")
ax[0].set_ylabel("Magnitud")
ax[0].grid(True)

# Señal reconstruida
ax[1].plot(t, [signal_function(tt, T) for tt in t], label="Señal Original", linestyle="--", alpha=0.7)
ax[1].plot(t, reconstructed_signal, label="Señal Reconstruida")
ax[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
ax[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
ax[1].set_title("Señal Reconstruida")
ax[1].set_xlabel("t")
ax[1].set_ylabel("x(t)")
ax[1].legend()
ax[1].grid(True)

# Mostrar gráficos
st.pyplot(fig)
