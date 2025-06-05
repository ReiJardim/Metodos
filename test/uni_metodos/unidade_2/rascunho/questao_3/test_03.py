# %% [markdown]
# # Questão 3: Método de Newton
#
# **Considere os pontos: A(0,0), B(0.1, 0.09983), C(0.2, 0.19867), D(0.3, 0.29552) e E(0.4, 0.38941), referentes à função $f(x) = \sin(x)$.**
#
# **a) Interpole $\sin(0.0625)$ por meio de uma polinomial interpoladora de quarta ordem.**
# **b) Interpole $\sin(0.25)$ por meio de uma polinomial interpoladora quadrática.**
#
# **Anotações Importantes:**
# * **i) Escreva aqui o valor encontrado com precisão de 4 casas de $\sin(0.0625)$:** _______
# * **ii) Escreva aqui o valor encontrado com precisão de 4 casas de $\sin(0.25)$:** _______
# * **iii) Preencher na tabela os valores encontrados para as diferenças divididas.**
# * **v) Delimite o erro de interpolação do item (b). Neste caso o erro da derivada com precisão de 5 casas decimais.**

# %% [markdown]
# ## Importações e Dados Iniciais

# %%
import math
import numpy as np
import pandas as pd  # Para exibir a tabela de diferenças divididas

# %% [markdown]
# ### Dados do Problema
# Pontos fornecidos $(x_i, f(x_i))$ onde $f(x_i) = \sin(x_i)$.

# %%
x_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
y_points = np.array([0.00000, 0.09983, 0.19867, 0.29552, 0.38941])

# Número de pontos
n_total_points = len(x_points)

# %% [markdown]
# ## iii) Tabela de Diferenças Divididas
#
# A tabela de diferenças divididas é fundamental para o Método de Newton.
# $f[x_i]$ são os próprios $f(x_i)$.
# $f[x_i, x_{j}] = \frac{f[x_j] - f[x_i]}{x_j - x_i}$
# $f[x_i, x_{j}, x_{k}] = \frac{f[x_j, x_k] - f[x_i, x_j]}{x_k - x_i}$
# E assim por diante.
#
# A tabela solicitada tem as seguintes colunas para as diferenças divididas:
# * $f[]$: Diferenças divididas de 1ª ordem (e.g., $f[x_i, x_{i+1}]$)
# * $f[,]$: Diferenças divididas de 2ª ordem (e.g., $f[x_i, x_{i+1}, x_{i+2}]$)
# * $f[,,]$: Diferenças divididas de 3ª ordem (e.g., $f[x_i, x_{i+1}, x_{i+2}, x_{i+3}]$)
# * $f[,,,]$: Diferenças divididas de 4ª ordem (e.g., $f[x_i, x_{i+1}, x_{i+2}, x_{i+3}, x_{i+4}]$)

# %%
# Inicializar a matriz de diferenças divididas
# A primeira coluna (índice 0) são os y_points (f(x_i))
divided_diff_table = np.zeros((n_total_points, n_total_points))
divided_diff_table[:, 0] = y_points

# Calcular as diferenças divididas
for j in range(1, n_total_points):  # Coluna da tabela de diferenças (ordem da diferença)
    for i in range(n_total_points - j):  # Linha da tabela de diferenças
        divided_diff_table[i, j] = \
            (divided_diff_table[i+1, j-1] - divided_diff_table[i, j-1]) / \
            (x_points[i+j] - x_points[i])

# Preparar a tabela para exibição conforme o formato da questão
# Colunas: i, x_i, f(x_i), f[], f[,], f[,,], f[,,,]
# Os valores nas colunas de diferença são populados de cima para baixo.
# f[] (Ordem 1): divided_diff_table[i, 1] para i de 0 a n-2
# f[,] (Ordem 2): divided_diff_table[i, 2] para i de 0 a n-3
# ...

df_display = pd.DataFrame({
    'i': range(n_total_points),
    'x_i': x_points,
    'f(x_i)': y_points
})

# Adicionando colunas de diferenças divididas ao DataFrame para exibição
# As colunas da questão f[], f[,], etc. são as divided_diff_table[0:, 1], divided_diff_table[0:, 2] etc.
# mas precisamos alinhar corretamente (mostrar NaN onde não aplicável)
col_labels = ['f[]', 'f[,]', 'f[,,]', 'f[,,,]']
for order in range(1, n_total_points):  # Ordem da diferença (1 a 4)
    col_name = col_labels[order-1]
    # Criar uma série com NaNs e preencher os valores calculados
    diff_values = [np.nan] * n_total_points
    for i in range(n_total_points - order):
        diff_values[i] = divided_diff_table[i, order]
    df_display[col_name] = diff_values


print("Tabela de Diferenças Divididas (iii):")
print(df_display.to_string(formatters={
    'x_i': '{:.1f}'.format,
    'f(x_i)': '{:.5f}'.format,
    'f[]': lambda x: f'{x:.5f}' if not pd.isna(x) else '',
    'f[,]': lambda x: f'{x:.5f}' if not pd.isna(x) else '',
    'f[,,]': lambda x: f'{x:.5f}' if not pd.isna(x) else '',
    'f[,,,]': lambda x: f'{x:.5f}' if not pd.isna(x) else ''
}))

# Os coeficientes do polinômio de Newton P_n(x) são a primeira linha da tabela de diferenças divididas:
# f[x_0], f[x_0,x_1], f[x_0,x_1,x_2], ...
# Estes são divided_diff_table[0,0], divided_diff_table[0,1], divided_diff_table[0,2], ...
newton_coeffs_all = divided_diff_table[0, :]


# %% [markdown]
# ## Polinômio Interpolador de Newton
#
# O polinômio de Newton é dado por:
# $P_n(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + \dots + f[x_0,\dots,x_n](x-x_0)\dots(x-x_{n-1})$
#
# Os coeficientes $f[x_0], f[x_0,x_1], \dots$ são os elementos da primeira linha da tabela de diferenças divididas.

# %%
def newton_polynomial(x_val, x_data, coeffs, order):
    """
    Calcula o valor do polinômio interpolador de Newton.
    x_val: O ponto onde o polinômio é avaliado.
    x_data: Array dos valores de x_i usados para construir o polinômio.
    coeffs: Array dos coeficientes de Newton (f[x_0], f[x_0,x_1], ...).
    order: A ordem do polinômio (n). O número de coeficientes será order+1.
    """
    result = coeffs[0]
    term = 1.0
    for i in range(1, order + 1):  # order+1 termos no total
        term *= (x_val - x_data[i-1])
        result += coeffs[i] * term
    return result

# %% [markdown]
# ## a) Interpolar $\sin(0.0625)$ (Quarta Ordem)
#
# Usaremos todos os 5 pontos (A, B, C, D, E) para um polinômio de 4ª ordem ($P_4(x)$).
# Os coeficientes são $f[x_0], f[x_0,x_1], f[x_0,x_1,x_2], f[x_0,x_1,x_2,x_3], f[x_0,x_1,x_2,x_3,x_4]$.
# Estes são: `newton_coeffs_all[0]` a `newton_coeffs_all[4]`.


# %%
x_eval_a = 0.0625
order_a = 4  # Quarta ordem

# Os x_points para P4 são todos os x_points
# Os coeficientes para P4 são os 5 primeiros da diagonal da tabela (newton_coeffs_all[0] a newton_coeffs_all[4])
p4_00625 = newton_polynomial(x_eval_a, x_points, newton_coeffs_all, order_a)

print(f"Polinômio de 4ª ordem P₄({x_eval_a}) = {p4_00625}")

# %% [markdown]
# ### i) Valor de $\sin(0.0625)$ com precisão de 4 casas decimais

# %%
# Arredondando para 4 casas decimais
p4_00625_rounded = round(p4_00625, 4)
print(
    f"i) Valor encontrado para sin(0.0625) (P₄(0.0625)) com 4 casas decimais: {p4_00625_rounded}")

# Comparação com o valor real de sin(0.0625)
sin_00625_actual = math.sin(x_eval_a)
print(f"Valor real de sin({x_eval_a}): {sin_00625_actual:.7f}")

# %% [markdown]
# ## b) Interpolar $\sin(0.25)$ (Quadrática)
#
# Para um polinômio de 2ª ordem ($P_2(x)$), precisamos de 3 pontos.
# Para interpolar em $x=0.25$, escolheremos os 3 pontos mais próximos que "cercam" $0.25$.
# Pontos disponíveis: (0,0), (0.1, 0.09983), (0.2, 0.19867), (0.3, 0.29552), (0.4, 0.38941).
# Os pontos $x=0.2$ e $x=0.3$ são os mais próximos. Podemos escolher o conjunto {$x_1=0.1, x_2=0.2, x_3=0.3$} ou {$x_2=0.2, x_3=0.3, x_4=0.4$}.
# Vamos usar os pontos $x_B(0.1), x_C(0.2), x_D(0.3)$ como nossos $x'_0, x'_1, x'_2$.
# Os coeficientes do polinômio $P_2(x) = f[x'_0] + f[x'_0,x'_1](x-x'_0) + f[x'_0,x'_1,x'_2](x-x'_0)(x-x'_1)$ serão:
# * $f[x'_0] = f[x_1] = f(0.1) = 0.09983$
# * $f[x'_0,x'_1] = f[x_1,x_2]$ (da tabela: `divided_diff_table[1,1]`)
# * $f[x'_0,x'_1,x'_2] = f[x_1,x_2,x_3]$ (da tabela: `divided_diff_table[1,2]`)

# %%
x_eval_b = 0.25
order_b = 2  # Quadrática (2ª ordem)

# Pontos para P2: x_1=0.1, x_2=0.2, x_3=0.3
# Correspondem aos índices 1, 2, 3 na lista x_points
x_points_b = x_points[1:4]  # x_points_b será [0.1, 0.2, 0.3]

# Coeficientes para P2 usando os pontos x_points_b:
# f[x_1] = y_points[1]
# f[x_1, x_2] = divided_diff_table[1,1] (segundo elemento da 2ª coluna de DD)
# f[x_1, x_2, x_3] = divided_diff_table[1,2] (segundo elemento da 3ª coluna de DD)
coeffs_b = [
    divided_diff_table[1, 0],  # y_points[1] ou f[x_1]
    divided_diff_table[1, 1],  # f[x_1, x_2]
    divided_diff_table[1, 2]  # f[x_1, x_2, x_3]
]

print(f"Pontos x para P₂: {x_points_b}")
print(f"Coeficientes de Newton para P₂: {coeffs_b}")

p2_025 = newton_polynomial(x_eval_b, x_points_b, coeffs_b, order_b)

print(f"\nPolinômio de 2ª ordem P₂({x_eval_b}) = {p2_025}")


# %% [markdown]
# ### ii) Valor de $\sin(0.25)$ com precisão de 4 casas decimais

# %%
# Arredondando para 4 casas decimais
p2_025_rounded = round(p2_025, 4)
print(
    f"ii) Valor encontrado para sin(0.25) (P₂({x_eval_b})) com 4 casas decimais: {p2_025_rounded}")

# Comparação com o valor real de sin(0.25)
sin_025_actual = math.sin(x_eval_b)
print(f"Valor real de sin({x_eval_b}): {sin_025_actual:.7f}")

# %% [markdown]
# ## v) Delimitação do Erro para o Item (b)
#
# O erro para o polinômio interpolador de Newton de ordem $n$ é dado por:
# $E_n(x) = f[x_0, x_1, \dots, x_n, x] \prod_{i=0}^{n} (x-x_i)$
# Ou, usando a derivada:
# $E_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x-x_i)$
#
# Para o item (b), temos $n=2$ (polinômio quadrático). Os pontos usados foram $x'_0=0.1, x'_1=0.2, x'_2=0.3$.
# $E_2(x) = \frac{f'''(\xi)}{3!} (x-x'_0)(x-x'_1)(x-x'_2)$
#
# A função é $f(x) = \sin(x)$.
# $f'(x) = \cos(x)$
# $f''(x) = -\sin(x)$
# $f'''(x) = -\cos(x)$
#
# Precisamos encontrar o máximo de $|f'''(\xi)| = |-\cos(\xi)|$ no intervalo que contém os pontos de interpolação ($0.1, 0.2, 0.3$) e o ponto de avaliação ($x=0.25$).
# O intervalo é $[0.1, 0.3]$.
# A questão pede para usar o erro da derivada com precisão de 5 casas decimais.
# Em $[0.1, 0.3]$ radianos (aproximadamente $5.7^\circ$ a $17.2^\circ$), $\cos(\xi)$ é positivo e decrescente.
# Portanto, $\max_{\xi \in [0.1, 0.3]} |-\cos(\xi)| = |-\cos(0.1)| = \cos(0.1)$.
# $\cos(0.1) \approx 0.995004165$. Com 5 casas decimais: $0.99500$.

# %%
# Cálculo do termo do produto (x-x_i)
x_eval_b = 0.25  # ponto de avaliação
# x_points_b foram [0.1, 0.2, 0.3]
product_term_error = (x_eval_b - x_points_b[0]) * \
                     (x_eval_b - x_points_b[1]) * \
                     (x_eval_b - x_points_b[2])

print(f"Termo do produto (x-x'_0)(x-x'_1)(x-x'_2) = {product_term_error}")

# Máximo da terceira derivada
# f'''(xi) = -cos(xi). Queremos max |-cos(xi)| em [0.1, 0.3]
# max |-cos(xi)| = cos(0.1)
max_f_triple_prime_abs = math.cos(0.1)  # Valor exato
# Com 5 casas decimais, conforme solicitado
max_f_triple_prime_abs_approx = 0.99500

print(f"Valor de cos(0.1) ≈ {max_f_triple_prime_abs:.7f}")
print(
    f"Valor de M_3 = max|f'''(ξ)| (com 5 casas decimais) = {max_f_triple_prime_abs_approx}")

# Fatorial de (n+1) = 3!
n_plus_1_factorial = math.factorial(2 + 1)  # 3! = 6

# Delimitação do erro
error_bound_b = (max_f_triple_prime_abs_approx /
                 n_plus_1_factorial) * abs(product_term_error)

print(
    f"\nDelimitação do erro para P₂({x_eval_b}): |E₂({x_eval_b})| ≤ {error_bound_b:.7e}")

# Erro real para P₂(0.25)
actual_error_b = abs(sin_025_actual - p2_025)
print(f"Erro real |sin(0.25) - P₂(0.25)| = {actual_error_b:.7e}")
if actual_error_b <= error_bound_b:
    print("O erro real é menor ou igual à delimitação calculada.")
else:
    print("Atenção: O erro real é MAIOR que a delimitação calculada (verificar cálculos ou aproximações).")


# %% [markdown]
# ## Resumo das Respostas Solicitadas
#
# * **i) Valor de $\sin(0.0625)$ (P₄(0.0625)) com 4 casas decimais:**
# * **ii) Valor de $\sin(0.25)$ (P₂(0.25)) com 4 casas decimais:**
# * **iii) Tabela de Diferenças Divididas:** (Exibida acima)
# * **v) Delimitação do erro de interpolação do item (b):**

# %%
print("Anotações Importantes - Respostas:")
print(
    f"i) Escreva aqui o valor encontrado com precisão de 4 casas de sen (0.0625): {p4_00625_rounded}")
print(
    f"ii) Escreva aqui o valor encontrado com precisão de 4 casas de sen (0.25): {p2_025_rounded}")
print(f"iii) A tabela de diferenças divididas foi exibida na seção correspondente.")
print(
    f"v) Delimite o erro de interpolação do item (b): |E₂(0.25)| ≤ {error_bound_b:.3e} (usando M₃ ≈ {max_f_triple_prime_abs_approx})")
