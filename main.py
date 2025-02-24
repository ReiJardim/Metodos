import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Métodos Computacionais")


# Define the x values
x = np.linspace(0, 10, 400)  # From 0 to 10, with 400 points

variabel_a = st.slider("Variavel A", 0.0, 10.0, 0.1)
variabel_b = st.slider("Variavel B", 0.0, 10.0, 0.1)

# Define the function
y = np.exp(-x)*variabel_a - np.cos(x)*variabel_b

st.title(f"np.exp(-x)*{variabel_a} - np.cos(x)*{variabel_b}")

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
st.pyplot(fig)


tab1, tab2, tab3 = st.tabs(["Bissecção", "Secante", "Newton"])


with tab1:
    st.write("Conteúdo da aba Bissecção")

with tab2:
    st.write("Conteúdo da aba Secante")

with tab3:
    st.write("Conteúdo da aba Newton")
