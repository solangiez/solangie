import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.cm as cm
import time
import scipy.integrate
import imageio
import os
import plotly.graph_objects as go
txx=0
thh=0
x=0
h=0
delta= 0.1
# #####################################################################señales discretas

# función 1
n1 = np.arange(-5, 5)
xn1 = 6 - abs(n1)

# función 2
n2 = np.arange(-5, 5)
xn2 = np.ones(len(n2))

# función 3
n3 = np.arange(-2, 8)
xn3 = np.ones(len(n3))

# función 4
n4 = np.arange(-1, 9)
xn4 = (9 / 11) ** (n4)

# #####################################################################señales continuas
delta = 0.1
# primera señal
t1 = np.arange(0, 3, delta)
t2 = np.arange(3, 5 + delta, delta)

x1 = 2 * np.ones(len(t1))
x2 = -2 * np.ones(len(t2))
t_1 = np.concatenate((t1, t2))
x_t1 = np.concatenate((x1, x2))

tx1 = np.arange(0, 5 + delta, delta)

# segunda señal
t_2 = np.arange(-1, 1 + delta, delta)
x_t2 = -t_2
tx2 = np.arange(-1, 1 + delta, delta)

# tercera señal
t1 = np.arange(-1, 1 + delta, delta)
t2 = np.arange(1, 3 + delta, delta)
t3 = np.arange(3, 5 + delta, delta)

x1 = 2 * np.ones(len(t1))
x2 = 4 - 2 * t2
x3 = -2 * np.ones(len(t3))
t_3 = np.concatenate((t1, t2, t3))
x_t3 = np.concatenate((x1, x2, x3))

tx3 = np.arange(-1, 5 + delta, delta)

# cuarta señal
t1 = np.arange(-3, 0 + delta, delta)
t2 = np.arange(0, 3 + delta, delta)

x1 = np.exp(t1)
x2 = np.exp(-t2)
t_4 = np.concatenate((t1, t2))
x_t4 = np.concatenate((x1, x2))

tx4 = np.arange(-3, 3 + delta, delta)



##################### definir las funciones de los cálculos
################################ función para discretas
def convdis(x, h):
    n1 = np.arange(0, len(x))
    x_k = x
    hmk = np.flip(h)

    ini = min(n1) - len(h)
    fin = max(n1) + len(h)

    ejetao = np.arange(ini, fin + 1)
    ejeconv = np.zeros(len(ejetao))

    ejexn = np.concatenate((np.zeros(len(h)), x_k, np.zeros(len(h))))

    filenames = []  # Lista para guardar los nombres de las imágenes

    for tao in range(0, len(x) + len(h) - 1):
        thmk = np.concatenate((np.zeros(tao + 1), hmk, np.zeros(len(x) + len(h) - 1 - tao)))
        idx = tao + len(h) - 1
        ejeconv[idx] = np.dot(ejexn, thmk)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
        ax1.clear()
        ax1.stem(ejetao, ejexn, 'g', markerfmt='go', basefmt=" ")
        ax1.stem(ejetao, thmk, 'm', markerfmt='mo', basefmt=" ")
        ax1.stem(ejetao, ejeconv, 'b', markerfmt='bo', basefmt=" ")
        ax1.fill_between(ejetao, ejeconv, color='b', alpha=0.3)
        ax1.stem(ejetao[idx], 0, 'ro')

        ax2.clear()
        ax2.stem(ejetao, ejeconv, 'b', markerfmt='bo', basefmt=" ")
        ax2.stem(ejetao[idx], ejeconv[idx], 'ro')

        filename = f"frame_{tao}.png"
        plt.savefig(filename)
        filenames.append(filename)
          
        plt.close(fig)

    with imageio.get_writer('convolucion.gif', mode='I', duration=100) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    for filename in filenames:
     os.remove(filename)

    return 'convolucion.gif'

def con(x, h, t1, t2, delta):
    x_t = x
    hmk = np.flip(h)

    ini = min(t1) - (round(max(t2), 2) - min(t2))
    fin = round(max(t1)) + (round(max(t2), 2) - min(t2))

    ejetao = np.arange(ini, fin + delta, delta)
    ejeconv = np.zeros(len(ejetao))

    ejext = np.concatenate((np.zeros(len(h) - 1), x_t, np.zeros(len(h) - 1)))

    frames = []  # Lista para almacenar los frames de la animación
    
    for tao in range(0, len(x) - 1 + len(h) - 1):
        thmk = np.concatenate((np.zeros(tao), hmk, np.zeros(len(x_t) - 1 + len(h) - tao - 1)))
        prod = ejext * thmk
        ejeconv[tao] = scipy.integrate.simpson(prod, dx=delta)

        # Crear un frame para cada paso de la convolución
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=ejetao, y=ejext, mode='lines', name='x(t)', line=dict(color='green')),
                    go.Scatter(x=ejetao, y=thmk, mode='lines', name='h(t-τ)', line=dict(color='magenta')),
                    go.Scatter(x=ejetao, y=ejeconv, mode='lines', name='Convolución', line=dict(color='blue')),
                ],
                layout=go.Layout(title=f"Paso {tao+1}/{len(x) + len(h) - 2}")
            )
        )
    
    # Configuración del gráfico inicial
    fig = go.Figure(
        data=[
            go.Scatter(x=ejetao, y=ejext, mode='lines', name='x(t)', line=dict(color='green')),
            go.Scatter(x=ejetao, y=hmk, mode='lines', name='h(t-τ)', line=dict(color='magenta')),
            go.Scatter(x=ejetao, y=ejeconv, mode='lines', name='Convolución', line=dict(color='blue'))
        ],
        layout=go.Layout(
            title="Animación de la convolución de señales",
            xaxis_title="Tiempo",
            yaxis_title="Amplitud",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')])]
            )]
        ),
        frames=frames
    )

    return fig
# Llamada a la función
#con(x_t1, x_t2, tx1, tx2, 0.1)
#convdis(xn1, xn2)




st.title("convolución de señales continuas y discretas")

tiposen=st.sidebar.selectbox("ingrese el tipo de señal que quiere transformar" , ["seleccione una opcion" ,"señal discreta", "señal continua"])

if tiposen=="señal discreta":
 opcion=st.sidebar.selectbox("cual conjunto de señales quiere transformar" ,[ "selecccione","señales a", "señales b"])
 if opcion=="señales a":
  fig1, axs = plt.subplots(1, 2, figsize=(5,3 ))

# Gráfico 1
  axs[ 0].stem(n1, xn1)
  axs[0].set_title('señal x_n')

# Gráfico 2
  axs[1].stem(n2, xn2)
  axs[1].set_title('señal h_n')

# Ajustar el layout
  plt.tight_layout()

# Mostrar el gráfico en Streamlit
  st.pyplot(fig1) 
  opcion2=st.sidebar.selectbox("cual es la señal quiere que se mueva en el proceso" ,["seleccione", "x_n", "h_n"])
  if opcion2=="x_n":
    gif_filename = convdis(xn2, xn1)

    st.image(gif_filename, caption="Convolución animada de señales discretas", use_column_width=True)
  elif opcion2=="h_n":
    gif_filename = convdis(xn1, xn2)

    st.image(gif_filename, caption="Convolución animada de señales discretas", use_column_width=True)
 elif opcion=="señales b":

# Crear figura y subplots
  fig2, axs = plt.subplots(1, 2, figsize=(5, 3))


# Gráfico 3
  axs[0].stem(n3, xn3)
  axs[0].set_title('Función x_n')

# Gráfico 4
  axs[1].stem(n4, xn4)
  axs[1].set_title("h_n")

# Ajustar el layout
  plt.tight_layout()

# Mostrar el gráfico en Streamlit
  st.pyplot(fig2)

  opcion2=st.sidebar.selectbox("cual es la señal quiere que se mueva en el proceso" ,["seleccione" "x_n", "h_n"])
  if opcion2=="x_n":
    gif_filename = convdis(xn4, xn3)

    st.image(gif_filename, caption="Convolución animada de señales continuas", use_column_width=True)
  elif opcion2=="h_n":
    gif_filename = convdis(xn3, xn4)

    st.image(gif_filename, caption="Convolución animada de señales continuas", use_column_width=True)
elif tiposen=="señal continua":
 senx=st.sidebar.selectbox("cual señal va a asignar como x_n" ,[ "seleccione una opcion","señal a", "señal b" , "señal c" , "señal d"])
 senh=st.sidebar.selectbox("cual señal va a asignar como h_n" ,[ "seleccione una opcion","señal a", "señal b" , "señal c" , "señal d"])
 ############################################################# para x
 if senx=="señal a":
  x=x_t1
  tx=tx1
  txx=t_1
 if senx=="señal b":
  x=x_t2
  tx=tx2
  txx=t_2
 if senx=="señal c":
  x=x_t3
  tx=tx3
  txx=t_3
 if senx=="señal d":
  x=x_t4
  tx=tx4
  txx=t_4
  ### para h
 if senh=="señal a":
  h= x_t1
  th= tx1
  thh=t_1
 if senh=="señal b":
  h=x_t2
  th=tx2
  thh=t_2
 if senh=="señal c":
  h=x_t3
  th=tx3
  thh=t_3
 if senh=="señal d":
  h=x_t4
  th=tx4
  thh=t_4
 
 if (senx !="seleccione una opcion") and (senh !="seleccione una opcion"):
  fig2, axs = plt.subplots(1, 2, figsize=(5, 3))


# Gráfico 3
  axs[0].plot(txx, x)
  axs[0].set_title('Función x_t')

# Gráfico 4
  axs[1].plot(thh, h)
  axs[1].set_title("h_t")

# Ajustar el layout
  plt.tight_layout()

# Mostrar el gráfico en Streamlit
  st.pyplot(fig2)
  seninv=st.sidebar.selectbox("cual señal se va a mover en la convolucion" ,[ "seleciona una opcion", "x_n" , "h_n" ])
  if seninv=="x_n":
   fig =con(h, x, th, tx, 0.1)
   
   #st.image(gif_filename, caption="Convolución animada de señales continuas", use_column_width=True)
   st.plotly_chart(fig)
  elif seninv=="h_n":
   fig =  con(x, h, tx, th, 0.1)

   #st.image(gif_filename, caption="Convolución animada de señales continuas", use_column_width=True)   
 


   


