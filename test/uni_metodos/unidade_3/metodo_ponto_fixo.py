import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Para formatação dos eixos, se necessário

# Exemplo de valores (pode ser substituído por input do usuário)
f_str_exemplo = str('2*x*y')
x_exemplo = 1
y_exemplo = 2
h_exemplo = 0.1
# Ajustando o número de iterações para corresponder à imagem de exemplo, se desejado.
# A imagem mostra 12 iterações após o ponto inicial (Iter. 0 a Iter. 12)
interacoes_exemplo = 12


def resolver_edo_ponto_medio(f_str, x_inicial, y_inicial, h_passo, num_total_iteracoes):
    """
    Resolve uma EDO de primeira ordem dy/dx = f(x,y) usando o método do Ponto Médio.

    Args:
        f_str (str): A equação dy/dx = f(x,y) como string.
        x_inicial (float/int): Valor inicial de x (x0).
        y_inicial (float/int): Valor inicial de y (y0).
        h_passo (float/int): Tamanho do passo (h).
        num_total_iteracoes (int): Número de iterações a serem calculadas APÓS o ponto inicial.
                                     (Ex: 10 iterações resultará em 11 linhas, de Iter. 0 a Iter. 10)

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
        x0_global = float(x_inicial)
        y0_global = float(y_inicial)
        h = float(h_passo)
        num_iteracoes = int(num_total_iteracoes)
    except ValueError:
        error_msg = "Erro: Entrada inválida. Certifique-se de que os valores numéricos (x_inicial, y_inicial, h_passo, num_total_iteracoes) estão corretos."
        return None, error_msg, None, None, None

    # Tenta criar a função a partir da string de entrada
    try:
        allowed_names = {"x": None, "y": None, "math": math}

        def func(x_eval, y_eval):
            return eval(f_str, {"__builtins__": {}}, {**allowed_names, "x": x_eval, "y": y_eval})
        # Teste inicial da função
        func(x0_global, y0_global)
    except Exception as e:
        error_msg = f"Erro ao interpretar a função f(x,y) ('{f_str}'): {e}\n" \
            "Certifique-se de que a equação usa 'x' e 'y' como variáveis.\n" \
            "Para funções matemáticas, use o prefixo 'math.', por exemplo, 'math.sin(x)'."
        return None, error_msg, None, None, None

    # Preparando as strings de informação para retorno
    equacao_info = f"Equação fornecida: dy/dx = {f_str}"
    condicoes_info = f"Condições iniciais: x0 = {x0_global}, y0 = {y0_global}"
    passo_info = f"Passo (h): {h}"
    # O número de iterações é o número de *passos* após o inicial.
    # Se num_iteracoes = 10, teremos Iter. 0, Iter. 1, ..., Iter. 10 (11 linhas)
    num_iteracoes_info = f"Número de iterações após o ponto inicial: {num_iteracoes}"

    # Inicialização das listas para armazenar os dados para o DataFrame
    # Estas listas vão armazenar os valores de x_i, y_i, k1_i, k2_i, erro_i para cada linha da tabela
    iter_col = []
    x_col = []  # Coluna 'x0' da imagem, representa o x no início do intervalo i
    y_col = []  # Coluna 'y0' da imagem, representa o y no início do intervalo i
    k1_col = []
    k2_col = []
    erro_col = []

    # Valores atuais para o loop
    x_i = x0_global
    y_i = y0_global

    # Iteração 0 (ponto inicial)
    iter_col.append(0)
    x_col.append(x_i)
    y_col.append(y_i)

    k1_val = func(x_i, y_i)
    k1_col.append(k1_val)

    k2_val = func(x_i + h / 2, y_i + (h / 2) * k1_val)
    k2_col.append(k2_val)
    erro_col.append(float('nan'))  # Erro não definido para a primeira iteração

    # Loop para as iterações seguintes
    # Se num_total_iteracoes = N, o loop vai de 1 até N.
    # Isso resultará em N+1 linhas no total (Iter. 0 até Iter. N)
    for i in range(1, num_iteracoes + 1):
        y_anterior = y_i  # y_i da iteração anterior (y_{n-1})

        # y_{i+1} = y_i + h * k2 (onde k2 foi calculado usando x_i, y_i da iteração anterior)
        # No nosso caso, k2_val é o k2 da linha anterior da tabela
        y_i = y_i + h * k2_val  # Este é o novo y_{n}

        # x_i é atualizado
        x_i = x0_global + i * h  # Este é o novo x_{n}

        iter_col.append(i)
        x_col.append(x_i)
        y_col.append(y_i)  # y_i calculado para esta iteração

        # Calcula k1 e k2 para ESTA iteração (x_i, y_i atuais)
        # Estes k1, k2 serão usados para calcular o y da PRÓXIMA iteração
        k1_val = func(x_i, y_i)
        k1_col.append(k1_val)

        k2_val = func(x_i + h / 2, y_i + (h / 2) * k1_val)
        k2_col.append(k2_val)

        erro_val = abs(y_i - y_anterior)
        erro_col.append(erro_val)

    # Criação do DataFrame do Pandas
    # As colunas são nomeadas para corresponder à imagem fornecida
    df = pd.DataFrame({
        'Iter.': iter_col,
        'x0': x_col,
        'y0': y_col,
        'K1': k1_col,
        'K2': k2_col,
        'ERRO |yn-yn-1|': erro_col
    })

    return df, equacao_info, condicoes_info, passo_info, num_iteracoes_info


def gerar_grafico_edo_ponto_medio(df_resultados, titulo_grafico="Solução da EDO pelo Método do Ponto Médio"):
    """
    Gera e exibe um gráfico da solução da EDO (y0 vs x0) e do erro
    (|yn - y_{n-1}| vs x0) a partir de um DataFrame.

    Args:
        df_resultados (pd.DataFrame): DataFrame com os resultados das iterações.
                                      Deve conter as colunas 'x0', 'y0', e 'ERRO |yn-yn-1|'.
        titulo_grafico (str, optional): Título do gráfico.
    """
    if not isinstance(df_resultados, pd.DataFrame):
        print("Erro: A entrada deve ser um DataFrame do Pandas.")
        return
    # Ajuste nos nomes das colunas para corresponder ao DataFrame do Ponto Médio
    required_cols = {'x0', 'y0', 'ERRO |yn-yn-1|'}
    if not required_cols.issubset(df_resultados.columns):
        print(
            f"Erro: O DataFrame deve conter as colunas {required_cols}.")
        return
    if df_resultados.empty:
        print("Erro: O DataFrame está vazio, nada para plotar.")
        return

    plt.figure(figsize=(12, 7))

    # 1. Plota y0 versus x0 (a solução principal)
    plt.plot(df_resultados['x0'], df_resultados['y0'], marker='o',
             linestyle='-', color='green', label='$y_n$ (Solução Ponto Médio)')

    # 2. Plota |yn - y_{n-1}| versus x0
    plt.plot(df_resultados['x0'], df_resultados['ERRO |yn-yn-1|'], marker='s',
             linestyle='--', color='purple', label='$|y_n - y_{n-1}|$ (Erro Aproximado)')

    plt.title(titulo_grafico, fontsize=16)
    plt.xlabel("$x_n$", fontsize=14)
    plt.ylabel("Valores", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    # plt.show() # Descomente para exibir o gráfico automaticamente
    return plt


# --- Exemplo de uso do método do Ponto Médio ---
dataframe_pm, eq_info_pm, ci_info_pm, p_info_pm, ni_info_pm = resolver_edo_ponto_medio(
    f_str_exemplo,
    x_exemplo,
    y_exemplo,
    h_exemplo,
    interacoes_exemplo
)

# Verificando se a função retornou com sucesso
if dataframe_pm is not None:
    print("Solução de EDO pelo Método do Ponto Médio")
    print("-----------------------------------------")
    print(eq_info_pm)
    print(ci_info_pm)
    print(p_info_pm)
    print(ni_info_pm)
    print("-----------------------------------------\n")
    print("Resultados das Iterações:")

    # Configurando o formato de exibição do Pandas para corresponder à imagem
    # A imagem mostra: x0 (3 casas), y0 (4 casas), K1 (2 casas), K2 (2 casas), ERRO (4 casas)
    # Usaremos to_string com formatação para melhor controle visual.
    formatters = {
        'x0': '{:.4f}'.format,
        'y0': '{:.4f}'.format,
        # A imagem parece usar 2 casas para K1, ex: 4.00, 10.29
        'K1': '{:.4f}'.format,
        # A imagem parece usar 2 casas para K2, ex: 4.62, 12.07
        'K2': '{:.4f}'.format,
        'ERRO |yn-yn-1|': '{:.4f}'.format
    }
    # Para K1 e K2, a imagem arredonda para 2 casas. O cálculo pode ter mais.
    # Se quisermos replicar EXATAMENTE a imagem, precisaríamos arredondar OS VALORES
    # antes de adicionar ao dataframe, ou aceitar que o print formatado é uma representação.
    # A imagem parece ter K1 e K2 com 2 ou 1 casa decimal se a segunda for zero.
    # Ex: K1=4, K2=4,62 na Iter. 0.
    # Para simplificar, vamos usar um número fixo de casas decimais no print,
    # mas os cálculos usarão a precisão total do float.

    # Replicando o formato da imagem, que tem um número variável de casas para K1 e K2
    # O mais próximo que podemos chegar com formatters é um número fixo.
    # A imagem parece usar: K1 (1-2 casas), K2 (2 casas)
    # No entanto, K1=10.29 (2 casas), K2=12.07 (2 casas) para Iter. 3
    # K1=14.46, K2=17.07 para Iter. 4
    # K1=4, K2=4.62 para Iter. 0.
    # Parece que K1 e K2 são geralmente com 2 casas decimais, mas K1=4 na Iter.0 é uma exceção.
    # Vamos usar 2 casas para K1 e K2 no display, como regra geral.

    # O to_string com float_format global pode ser mais simples
    # pd.options.display.float_format = '{:.4f}'.format
    # print(dataframe_pm.to_string(index=False, na_rep='NaN'))

    # Usando formatters para controle por coluna:
    print(dataframe_pm.to_string(index=False, na_rep='NaN', formatters=formatters))

    # Gerar e exibir o gráfico
    grafico_pm = gerar_grafico_edo_ponto_medio(
        dataframe_pm, titulo_grafico=f"Solução de {f_str_exemplo} pelo Método do Ponto Médio"
    )
    # Para exibir o gráfico em ambientes que não o fazem automaticamente (ex: scripts):
    # grafico_pm.show() # Descomente se quiser que o script pause e mostre o gráfico
    print("\nGráfico gerado (descomente plt.show() na função ou aqui para exibir).")


else:
    # 'eq_info_pm' conteria a mensagem de erro neste caso
    print(f"Não foi possível calcular a solução: {eq_info_pm}")

print("\n--- Exemplo com erro na equação (Ponto Médio) ---")
dataframe_erro_pm, msg_erro_pm, _, _, _ = resolver_edo_ponto_medio(
    "2*x*z",  # Equação com variável 'z' não definida
    1, 2, 0.1, 5
)
if dataframe_erro_pm is None:
    print(msg_erro_pm)
