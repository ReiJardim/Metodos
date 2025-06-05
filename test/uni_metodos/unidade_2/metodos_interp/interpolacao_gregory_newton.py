import numpy as np
import math


def calcular_diferencas_finitas(y):
    """
    Calcula a tabela de diferenças finitas.

    Args:
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.

    Returns:
        Uma lista contendo os coeficientes das diferenças finitas
        (Δ⁰y₀, Δ¹y₀, Δ²y₀, ...), que são os coeficientes do polinômio
        de Gregory-Newton.
    """
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = coef[i+1, j-1] - coef[i, j-1]

    return coef[0, :]  # Retorna Δ⁰y₀, Δ¹y₀, Δ²y₀, ...


def expandir_polinomio_gregory_newton(coef, x0, h):
    """
    Expande o polinômio de Gregory-Newton para a forma padrão ax² + bx + c.

    Args:
        coef: Coeficientes do polinômio (diferenças finitas).
        x0: Primeiro valor de x.
        h: Tamanho do passo.

    Returns:
        Coeficientes do polinômio na forma padrão [a, b, c].
    """
    n = len(coef)
    if n < 3:
        raise ValueError(
            "Precisa de pelo menos 3 pontos para expandir o polinômio")

    # Coeficientes do polinômio na forma padrão
    a = coef[2] / (2 * h**2)
    b = (coef[1] / h) - (coef[2] * (2*x0 + h)) / (2 * h**2)
    c = coef[0] - (coef[1] * x0 / h) + (coef[2] * x0 * (x0 + h)) / (2 * h**2)

    return [a, b, c]


def formatar_polinomio_latex(coef, h):
    """
    Formata o polinômio em LaTeX.

    Args:
        coef: Coeficientes do polinômio (diferenças finitas).
        h: Tamanho do passo (diferença entre x's consecutivos).

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

        # Adiciona o fator (u - 0)(u - 1)...(u - (i-1)) ao produto acumulado
        termo_novo = ""
        for j in range(i):
            if j == 0:
                termo_novo = f"u"
            else:
                termo_novo = f"{termo_novo}(u - {j})"

        termos_produto = termo_novo

        # Adiciona o termo completo ao polinômio string
        sinal = " + " if coef[i] >= 0 else " - "
        fatorial = math.factorial(i)
        latex_str += f"{sinal}\\frac{{{abs(coef[i]):.4f}}}{{{fatorial}}}{termos_produto}"

    # Substitui ' + -' por ' - ' para melhor formatação
    latex_str = latex_str.replace(" + -", " - ")

    # Adiciona o ambiente matemático do LaTeX
    return f"$P(x) = {latex_str}$"


def gregory_newton_interpolation(x, y, x_interp):
    """
    Realiza interpolação polinomial usando o método de Gregory-Newton e
    retorna o valor interpolado e as strings do polinômio.

    Args:
        x: Lista ou array numpy das coordenadas x dos pontos conhecidos.
        y: Lista ou array numpy das coordenadas y dos pontos conhecidos.
        x_interp: Coordenada x para a qual desejamos interpolar o valor y.

    Returns:
        tuple: (y_interpolado, polinomio_string, polinomio_latex, polinomio_padrao)
            - y_interpolado: O valor y interpolado correspondente a x_interp.
            - polinomio_string: Uma string representando o polinômio de Gregory-Newton.
            - polinomio_latex: Uma string representando o polinômio em LaTeX.
            - polinomio_padrao: Uma string representando o polinômio na forma padrão.
    """
    x = np.asarray(x)  # Garante que x é um array numpy
    y = np.asarray(y)  # Garante que y é um array numpy

    # Verifica se os pontos x são igualmente espaçados
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError(
            "Os pontos x devem ser igualmente espaçados para o método de Gregory-Newton")

    # 1. Calcular os coeficientes (diferenças finitas)
    coef = calcular_diferencas_finitas(y)
    n = len(coef)

    # 2. Calcular u = (x - x₀)/h
    u = (x_interp - x[0]) / h

    # 3. Avaliar o polinômio de Gregory-Newton em u
    y_interp = coef[0]
    termo = 1.0
    for i in range(1, n):
        termo *= (u - (i-1)) / i
        y_interp += coef[i] * termo

    # 4. Construir a string do polinômio
    poly_str = f"{coef[0]:.4f}"
    termos_produto = ""

    for i in range(1, n):
        if abs(coef[i]) < 1e-9:
            continue

        termo_novo = ""
        for j in range(i):
            if j == 0:
                termo_novo = f"u"
            else:
                termo_novo = f"{termo_novo}(u - {j})"

        termos_produto = termo_novo

        sinal = " + " if coef[i] >= 0 else " - "
        fatorial = math.factorial(i)
        poly_str += f"{sinal}{abs(coef[i]):.4f}/{fatorial}{termos_produto}"

    poly_str = poly_str.replace(" + -", " - ")

    # 5. Gerar a versão LaTeX do polinômio
    latex_str = formatar_polinomio_latex(coef, h)

    # 6. Expandir para a forma padrão
    a, b, c = expandir_polinomio_gregory_newton(coef, x[0], h)
    polinomio_padrao = f"{a:.4f}x² + {b:.4f}x + {c:.4f}"

    return y_interp, f"P(x) = {poly_str}", latex_str, polinomio_padrao


# --- Exemplo de Uso ---
# Pontos conhecidos (x, y) - devem ser igualmente espaçados
x_conhecidos = [0.3, 0.5, 0.7]
y_conhecidos = [1.5678, 3.6789, 8.8900]

# Ponto onde queremos interpolar
x_para_interpolar = 0.45

# Realizar a interpolação
y_interpolado, polinomio_final, polinomio_latex, polinomio_padrao = gregory_newton_interpolation(
    x_conhecidos, y_conhecidos, x_para_interpolar)

print(f"Os pontos conhecidos são:")
for i in range(len(x_conhecidos)):
    print(f"  ({x_conhecidos[i]}, {y_conhecidos[i]})")

print(f"\nO polinômio interpolador de Gregory-Newton é:")
print(f"  {polinomio_final}")

print(f"\nO polinômio em LaTeX é:")
print(f"  {polinomio_latex}")

print(f"\nO polinômio na forma padrão é:")
print(f"  P(x) = {polinomio_padrao}")

print(f"\nSubstituindo x = {x_para_interpolar} no polinômio, obtemos:")
print(f"  P({x_para_interpolar}) = {y_interpolado:.4f}")
