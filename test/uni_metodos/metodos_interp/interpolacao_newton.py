import numpy as np


def calcular_diferencas_divididas(x, y):
    """
    Calcula a tabela de diferenças divididas de Newton.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.

    Returns:
        Uma lista contendo os coeficientes da diagonal da tabela de
        diferenças divididas (f[x_0], f[x_0, x_1], f[x_0, x_1, x_2], ...),
        que são os coeficientes do polinômio de Newton.
    """
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])

    return coef[0, :]  # Retorna f[x_0], f[x_0, x_1], ...


def formatar_polinomio_latex(coef, x):
    """
    Formata o polinômio em LaTeX.

    Args:
        coef: Coeficientes do polinômio.
        x: Valores de x dos pontos conhecidos.

    Returns:
        String representando o polinômio em LaTeX.
    """
    # Começa com o primeiro coeficiente (termo constante)
    latex_str = f"{coef[0]:.4f}"
    termos_produto = ""

    for i in range(1, len(coef)):
        # Ignora termos com coeficiente muito próximo de zero
        if abs(coef[i]) < 1e-9:
            continue

        # Adiciona o fator (x - x_{i-1}) ao produto acumulado
        xi_val = x[i-1]
        if xi_val == 0:
            termo_novo = "x"
        elif xi_val > 0:
            termo_novo = f"(x - {xi_val:g})"
        else:  # xi_val < 0
            termo_novo = f"(x + {-xi_val:g})"

        termos_produto += termo_novo

        # Adiciona o termo completo ao polinômio string
        sinal = " + " if coef[i] >= 0 else " - "
        latex_str += f"{sinal}{abs(coef[i]):.4f}{termos_produto}"

    # Substitui ' + -' por ' - ' para melhor formatação
    latex_str = latex_str.replace(" + -", " - ")

    # Adiciona o ambiente matemático do LaTeX
    return f"$P(x) = {latex_str}$"


def newton_interpolation(x, y, x_interp):
    """
    Realiza interpolação polinomial usando o método de Newton e
    retorna o valor interpolado e as strings do polinômio.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.
        x_interp: Coordenada x para a qual desejamos interpolar o valor y.

    Returns:
        tuple: (y_interpolado, polinomio_string, polinomio_latex)
            - y_interpolado: O valor y interpolado correspondente a x_interp.
            - polinomio_string: Uma string representando o polinômio de Newton.
            - polinomio_latex: Uma string representando o polinômio em LaTeX.
    """
    x = np.asarray(x)  # Garante que x é um array numpy
    y = np.asarray(y)  # Garante que y é um array numpy

    # 1. Calcular os coeficientes (diferenças divididas)
    coef = calcular_diferencas_divididas(x, y)
    n = len(coef)

    # 2. Avaliar o polinômio de Newton em x_interp (Método de Horner)
    y_interp = coef[n-1]
    for i in range(n - 2, -1, -1):
        y_interp = y_interp * (x_interp - x[i]) + coef[i]

    # 3. Construir a string do polinômio
    # Começa com o primeiro coeficiente (termo constante)
    poly_str = f"{coef[0]:.4f}"
    termos_produto = ""

    for i in range(1, n):
        # Ignora termos com coeficiente muito próximo de zero
        if abs(coef[i]) < 1e-9:
            continue

        # Adiciona o fator (x - x_{i-1}) ao produto acumulado
        xi_val = x[i-1]
        if xi_val == 0:
            termo_novo = " * x"
        elif xi_val > 0:
            termo_novo = f" * (x - {xi_val:g})"
        else:  # xi_val < 0
            termo_novo = f" * (x + {-xi_val:g})"

        termos_produto += termo_novo

        # Adiciona o termo completo ao polinômio string
        sinal = " + " if coef[i] >= 0 else " - "
        poly_str += f"{sinal}{abs(coef[i]):.4f}{termos_produto}"

    # Substitui ' + -' por ' - ' para melhor formatação
    poly_str = poly_str.replace(" + -", " - ")

    # 4. Gerar a versão LaTeX do polinômio
    latex_str = formatar_polinomio_latex(coef, x)

    return y_interp, f"P(x) = {poly_str}", latex_str


# --- Exemplo de Uso ---
# Pontos conhecidos (x, y)
x_conhecidos = [1, 2, 3]
y_conhecidos = [3, 5, 12]

# Ponto onde queremos interpolar
x_para_interpolar = 1.5

# Realizar a interpolação
y_interpolado, polinomio_final, polinomio_latex = newton_interpolation(
    x_conhecidos, y_conhecidos, x_para_interpolar)

print(f"Os pontos conhecidos são:")
for i in range(len(x_conhecidos)):
    print(f"  ({x_conhecidos[i]}, {y_conhecidos[i]})")

print(f"\nO polinômio interpolador de Newton é:")
print(f"  {polinomio_final}")

print(f"\nO polinômio em LaTeX é:")
print(f"  {polinomio_latex}")

print(f"\nSubstituindo x = {x_para_interpolar} no polinômio, obtemos:")
print(f"  P({x_para_interpolar}) = {y_interpolado:.4f}")
