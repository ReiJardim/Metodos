import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Metodos",
    layout="wide",
    initial_sidebar_state="expanded",

)

st.title("Métodos Computacionais")

col01_01, col01_02 = st.columns([0.7, 0.3])

with col01_02:
    with st.container(border=True):
        st.title("Dados do Grafico")
        col01_02_01, col01_02_02 = st.columns([0.5, 0.5])

        with col01_02_01:
            x_inicial = st.number_input(
                "Valor inicial", value=0, placeholder="Digite um numero")
        with col01_02_02:
            x_final = st.number_input(
                "Valor final", value=10, placeholder="Digite um numero")
    padrao = True

    x0 = float(st.number_input("Digite a primeira estimativa inicial (x0): "))
    x1 = float(st.number_input("Digite a segunda estimativa inicial (x1): "))
    tol = float(st.number_input("Digite a tolerância desejada (ex: 1e-6): "))
    func_str = input(
        "Digite a função em termos de 'x' (ex: 'x*np.log(x) - 1'): ")

    def f(x): return eval(func_str)

x = np.linspace(x_inicial, x_final, 400)
y = x*np.log(x) - 1
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes objects

ax.plot(x, y, label='y = e^(-x) - cos(x)')

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Plot of y = e^(-x) - cos(x)')

# Add grid
ax.grid(True)

# Add legend
ax.legend()

# Show the plot using Streamlit
with col01_01:
    st.pyplot(fig)


tab1, tab2, tab3 = st.tabs(["Bissecção", "Secante", "Newton"])


with tab1:
    st.write("Conteúdo da aba Bissecção")

with tab2:
    st.write("Conteúdo da aba Secante")

with tab3:
    st.write("Conteúdo da aba Newton")
