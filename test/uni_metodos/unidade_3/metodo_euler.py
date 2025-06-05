import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Para formatação dos eixos, se necessário


# f_str = input(    "Digite a equação dy/dx = f(x,y) (ex: 2*x*y ou math.exp(x) - y): ")

f_str_exemplo = str('2*x*y')  # str() é redundante aqui, '2*x*y' já é string
x_exemplo = 1
y_exemplo = 2
h_exemplo = 0.1
interacoes_exemplo = 27


def resolver_edo_euler(f_str, x_inicial, y_inicial, h_passo, num_total_iteracoes):
    """
    Resolve uma EDO de primeira ordem dy/dx = f(x,y) usando o método de Euler.

    Args:
        f_str (str): A equação dy/dx = f(x,y) como string.
        x_inicial (float/int): Valor inicial de x (x0).
        y_inicial (float/int): Valor inicial de y (y0).
        h_passo (float/int): Tamanho do passo (h).
        num_total_iteracoes (int): Número de iterações a serem calculadas após o ponto inicial.

    Returns:
        tuple: Contendo:
            - df (pd.DataFrame): DataFrame com os resultados das iterações.
            - equacao_info (str): String descrevendo a equação fornecida.
            - condicoes_info (str): String descrevendo as condições iniciais.
            - passo_info (str): String descrevendo o passo h.
            - num_iteracoes_info (str): String descrevendo o número de iterações.
        Retorna (None, error_message, None, None, None) em caso de erro na conversão
        de input ou na interpretação da função.
    """

    try:
        x0 = float(x_inicial)
        y0 = float(y_inicial)
        h = float(h_passo)
        num_iteracoes = int(num_total_iteracoes)
    except ValueError:
        error_msg = "Erro: Entrada inválida. Certifique-se de que os valores numéricos (x_inicial, y_inicial, h_passo, num_total_iteracoes) estão corretos."
        return None, error_msg, None, None, None

    # Tenta criar a função a partir da string de entrada
    try:
        allowed_names = {"x": None, "y": None, "math": math}
        # Usamos f_str diretamente do parâmetro da função

        def func(x_eval, y_eval): return eval(f_str, {"__builtins__": {}}, {
            **allowed_names, "x": x_eval, "y": y_eval})
        # Teste inicial da função
        func(x0, y0)
    except Exception as e:
        error_msg = f"Erro ao interpretar a função f(x,y) ('{f_str}'): {e}\n" \
            "Certifique-se de que a equação usa 'x' e 'y' como variáveis.\n" \
            "Para funções matemáticas, use o prefixo 'math.', por exemplo, 'math.sin(x)'."
        return None, error_msg, None, None, None

    # Preparando as strings de informação para retorno
    equacao_info = f"Equação fornecida: dy/dx = {f_str}"
    condicoes_info = f"Condições iniciais: x0 = {x0}, y0 = {y0}"
    passo_info = f"Passo (h): {h}"
    num_iteracoes_info = f"Número de iterações: {num_iteracoes}"

    # Inicialização das listas para armazenar os dados
    resultados = []

    # Valores atuais
    x_k = x0
    y_k = y0
    y_anterior_para_erro = y0  # Usado para calcular |y_k - y_{k-1}|

    # Iteração 0 (ponto inicial)
    f_xy_k = func(x_k, y_k)
    erro_k = float('nan')
    resultados.append([0, x_k, y_k, f_xy_k, erro_k])

    # Loop para as iterações seguintes
    for i in range(1, num_iteracoes + 1):
        y_proximo = y_k + h * f_xy_k

        x_k = x0 + i * h
        y_k = y_proximo

        f_xy_k = func(x_k, y_k)
        erro_k = abs(y_k - y_anterior_para_erro)

        resultados.append([i, x_k, y_k, f_xy_k, erro_k])
        y_anterior_para_erro = y_k

    # Criação do DataFrame do Pandas
    colunas = ['Iter.', 'x_k', 'y_k', 'f(x_k,y_k)', '|y_k - y_{k-1}|']
    df = pd.DataFrame(resultados, columns=colunas)

    return df, equacao_info, condicoes_info, passo_info, num_iteracoes_info


def gerar_grafico_edo(df_resultados, titulo_grafico="Solução da EDO pelo Método de Euler"):
    """
    Gera e exibe um gráfico da solução da EDO (y_k vs x_k) e do erro
    (|y_k - y_{k-1}| vs x_k) a partir de um DataFrame.

    Args:
        df_resultados (pd.DataFrame): DataFrame com os resultados das iterações.
                                      Deve conter as colunas 'x_k', 'y_k', e '|y_k - y_{k-1}|'.
        titulo_grafico (str, optional): Título do gráfico.
                                         Default é "Solução da EDO pelo Método de Euler".
    """
    if not isinstance(df_resultados, pd.DataFrame):
        print("Erro: A entrada deve ser um DataFrame do Pandas.")
        return
    if not {'x_k', 'y_k', '|y_k - y_{k-1}|'}.issubset(df_resultados.columns):
        print(
            "Erro: O DataFrame deve conter as colunas 'x_k', 'y_k' e '|y_k - y_{k-1}|'.")
        return
    if df_resultados.empty:
        print("Erro: O DataFrame está vazio, nada para plotar.")
        return

    plt.figure(figsize=(12, 7))

    # 1. Plota y_k versus x_k (a solução principal)
    plt.plot(df_resultados['x_k'], df_resultados['y_k'], marker='o',
             linestyle='-', color='blue', label='$y_k$ (Solução de Euler)')

    # 2. Plota |y_k - y_{k-1}| versus x_k
    # A primeira entrada de erro é NaN, o matplotlib geralmente lida com isso
    # não plotando o ponto ou criando uma quebra na linha.
    plt.plot(df_resultados['x_k'], df_resultados['|y_k - y_{k-1}|'], marker='s',
             linestyle='--', color='red', label='$|y_k - y_{k-1}|$ (Erro)')

    # Adiciona títulos e legendas
    plt.title(titulo_grafico, fontsize=16)
    plt.xlabel("$x_k$", fontsize=14)
    # Rótulo genérico para o eixo y, já que plota y_k e o erro
    plt.ylabel("Valores", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Ajusta o layout para evitar que os rótulos se sobreponham
    plt.tight_layout()

    # Exibe o gráfico
    # plt.show()
    return plt


dataframe, eq_info, ci_info, p_info, ni_info = resolver_edo_euler(
    f_str_exemplo,
    x_exemplo,
    y_exemplo,
    h_exemplo,
    interacoes_exemplo
)

grafico = gerar_grafico_edo(
    dataframe, titulo_grafico="Exemplo de Gráfico da Solução da EDO")
# Verificando se a função retornou com sucesso
if dataframe is not None:
    print("Solução de EDO pelo Método de Euler")
    print("------------------------------------")
    print(eq_info)
    print(ci_info)
    print(p_info)
    print(ni_info)
    print("------------------------------------\n")
    print("Resultados das Iterações:")
    # Configura o Pandas para exibir floats com um número específico de casas decimais, se desejado
    # pd.options.display.float_format = '{:.5f}'.format # Exemplo para 5 casas decimais
    print(dataframe.to_string(index=False, na_rep='NaN'))
else:
    # 'eq_info' conteria a mensagem de erro neste caso
    print(f"Não foi possível calcular a solução: {eq_info}")

print("\n--- Exemplo com erro na equação ---")
dataframe_erro, msg_erro, _, _, _ = resolver_edo_euler(
    "2*x*z",  # Equação com variável 'z' não definida
    1, 2, 0.1, 5
)
if dataframe_erro is None:
    print(msg_erro)
