import numpy as np
import pandas as pd


def metodo_bissecao(f, a, b, erro):

    # Verifica se o intervalo inicial é válido
    if f(a) * f(b) >= 0:
        raise ValueError("O intervalo inicial não é válido.")

    # VARIÁVEIS
    iteracoes = 0
    resultados = []
    c_anterior = None  # Armazena o valor anterior de c para calcular o erro relativo

    while True:
        iteracoes += 1
        c = (a + b) / 2

        erro_absoluto = abs((b - a))
        # Calcula o erro relativo, tratando divisão por zero
        erro_relativo = abs(
            (c - c_anterior) / c) if c_anterior is not None and c != 0 else float('inf')

        resultados.append([iteracoes, a, b, c, erro_absoluto, erro_relativo])

        # Sempre atualiza
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

        # Sempre atualiza
        c_anterior = c

        # Erro tolerado
        if erro_absoluto < erro:
            break

    df = pd.DataFrame(resultados, columns=[
                      "Iteração", "a", "b", "c", "Erro Absoluto", "Erro Relativo"])
    return df, c, iteracoes
