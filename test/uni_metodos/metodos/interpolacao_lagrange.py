import numpy as np


def calcular_polinomio_lagrange(x, y):
    """
    Calcula o polinômio interpolador de Lagrange.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.

    Returns:
        Uma função que calcula o valor do polinômio interpolador em um ponto x.
    """
    n = len(x)

    def L(x_interp):
        resultado = 0
        for i in range(n):
            # Calcula o termo L_i(x)
            termo = y[i]
            for j in range(n):
                if j != i:
                    termo *= (x_interp - x[j]) / (x[i] - x[j])
            resultado += termo
        return resultado

    return L


def formatar_polinomio_latex(x, y):
    """
    Formata o polinômio de Lagrange em LaTeX.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.

    Returns:
        String representando o polinômio em LaTeX.
    """
    n = len(x)
    latex_str = ""

    for i in range(n):
        if i > 0:
            latex_str += " + "

        # Adiciona o coeficiente y[i]
        latex_str += f"{y[i]:.4f}"

        # Adiciona os termos (x - x_j)/(x_i - x_j) para j != i
        for j in range(n):
            if j != i:
                if x[j] == 0:
                    latex_str += f"x"
                elif x[j] > 0:
                    latex_str += f"(x - {x[j]:g})"
                else:
                    latex_str += f"(x + {-x[j]:g})"

                # Adiciona o denominador
                if x[i] - x[j] != 1:
                    latex_str += f"/{x[i] - x[j]:g}"

    return f"$P(x) = {latex_str}$"


def lagrange_interpolation(x, y, x_interp):
    """
    Realiza interpolação polinomial usando o método de Lagrange e
    retorna o valor interpolado e as strings do polinômio.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.
        x_interp: Coordenada x para a qual desejamos interpolar o valor y.

    Returns:
        tuple: (y_interpolado, polinomio_latex)
            - y_interpolado: O valor y interpolado correspondente a x_interp.
            - polinomio_latex: Uma string representando o polinômio em LaTeX.
    """
    x = np.asarray(x)  # Garante que x é um array numpy
    y = np.asarray(y)  # Garante que y é um array numpy

    # Verifica se há pontos duplicados
    if len(np.unique(x)) != len(x):
        raise ValueError("Não podem existir pontos x duplicados")

    # Calcula o polinômio interpolador
    P = calcular_polinomio_lagrange(x, y)

    # Avalia o polinômio no ponto desejado
    y_interp = P(x_interp)

    # Formata o polinômio em LaTeX
    latex_str = formatar_polinomio_latex(x, y)

    return y_interp, latex_str


# --- Exemplo de Uso ---
# Pontos conhecidos (x, y)
x_conhecidos = [-0.6321, 0.1065, 1]
y_conhecidos = [-1, -0.5, 0]

# Ponto onde queremos interpolar
x_para_interpolar = 0

# Realizar a interpolação
y_interpolado, polinomio_latex = lagrange_interpolation(
    x_conhecidos, y_conhecidos, x_para_interpolar)

print(f"Os pontos conhecidos são:")
for i in range(len(x_conhecidos)):
    print(f"  ({x_conhecidos[i]}, {y_conhecidos[i]})")

print(f"\nO polinômio interpolador de Lagrange é:")
print(f"  {polinomio_latex}")

print(f"\nSubstituindo x = {x_para_interpolar} no polinômio, obtemos:")
print(f"  P({x_para_interpolar}) = {y_interpolado:.4f}")

# Exemplo de uso do polinômio interpolador
P = calcular_polinomio_lagrange(x_conhecidos, y_conhecidos)
print("\nExemplos de valores do polinômio:")
for x in np.linspace(0.3, 0.7, 5):
    print(f"  P({x:.2f}) = {P(x):.4f}")
