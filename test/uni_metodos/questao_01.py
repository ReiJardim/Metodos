# %% [markdown]
# # Questão 1: Interpolação de Lagrange
#
# **Dado a tabela:**
#
# | x     | 0   | 0.1    | 0.2    | 0.3    | 0.4    | 0.5    |
# |-------|-----|--------|--------|--------|--------|--------|
# | $e^{3x}$ | 1   | 1.3499 | 1.8221 | 2.4596 | 3.3201 | 4.4817 |
#
# **Calcular $f(0.25)$, onde $f(x) = xe^{3x}$ usando polinômio de interpolação de 2º grau. Usar os valores sinalizados de negrito na tabela.**
#
# **Entrega:**
# * i) $f(0.25)$: _______
# * ii) $P_2(0.25)$: _______

# %% [markdown]
# ## Importações e Definições Iniciais

# %%
import math

# %% [markdown]
# ### Definição da Função $f(x)$ e Dados da Tabela
# A função a ser analisada é $f(x) = x \cdot e^{3x}$.
# Os valores de $e^{3x}$ são fornecidos na tabela.

# %%
# Definição da função f(x) = x * e^(3x)


def func_f(x_val):
    return x_val * math.exp(3 * x_val)


# Valores da tabela para e^(3x)
table_e_3x = {
    0.0: 1.0,
    0.1: 1.3499,
    0.2: 1.8221,
    0.3: 2.4596,
    0.4: 3.3201,
    0.5: 4.4817
}

# %% [markdown]
# ## Parte i): Calcular $f(0.25)$
# Esta é uma avaliação direta da função $f(x)$ em $x=0.25$.

# %%
x_target_direct = 0.25
f_025_calculated_direct = func_f(x_target_direct)

print(
    f"O valor calculado diretamente de f({x_target_direct}) é: {f_025_calculated_direct:.7f}")

# %% [markdown]
# ## Parte ii): Calcular $P_2(0.25)$ usando Polinômio de Interpolação de Lagrange de 2º Grau
#
# Para um polinômio de interpolação de 2º grau ($P_2(x)$), precisamos de 3 pontos $(x_0, y_0)$, $(x_1, y_1)$, e $(x_2, y_2)$.
# A questão instrui a "Usar os valores sinalizados de negrito na tabela". Como a imagem não permite uma identificação clara de quais valores estão em negrito, faremos uma suposição informada.
#
# Para interpolar em $x=0.25$, é comum escolher os pontos da tabela $(x_i, f(x_i))$ onde os $x_i$ estão mais próximos de $0.25$.
# Os valores de $f(x_i)$ serão calculados como $x_i \cdot (\text{valor de } e^{3x_i} \text{ da tabela})$.
#
# **Suposição:** Vamos assumir que os pontos $x$ "em negrito" (ou os mais apropriados para interpolar em $x=0.25$) são $x_0=0.2$, $x_1=0.3$, e $x_2=0.4$.
# Se outros pontos forem os corretos, a lista `x_coords_interp` abaixo deve ser alterada.

# %%
# Escolha dos pontos x para interpolação (suposição)
# Alternativa 1 (se os pontos negritos fossem 0.1, 0.2, 0.3): x_coords_interp = [0.1, 0.2, 0.3]
# Alternativa 2 (escolhida consistentemente)
x_coords_interp = [0.2, 0.3, 0.4]

# Verificar se os pontos escolhidos estão na tabela de e^(3x)
if not all(x_val in table_e_3x for x_val in x_coords_interp):
    print(
        f"Erro: Um ou mais dos x_coords_interp ({x_coords_interp}) não estão na tabela e^(3x) fornecida.")
    y_coords_interp = None  # Indica que não podemos prosseguir
else:
    # Calcular os valores de y = f(x) = x * e^(3x) para os pontos de interpolação,
    # usando os valores de e^(3x) da tabela.
    y_coords_interp = []
    print("Pontos (x_i, f(x_i)) usados para a interpolação de Lagrange:")
    for x_val in x_coords_interp:
        # f(x_val) = x_val * (valor de e^(3*x_val) da tabela)
        f_xi = x_val * table_e_3x[x_val]
        y_coords_interp.append(f_xi)
        print(f"  f({x_val}) = {x_val} * {table_e_3x[x_val]} = {f_xi:.5f}")

# %% [markdown]
# ### Função de Interpolação de Lagrange (2º Grau)
# O polinômio de Lagrange $P_2(x)$ é dado por:
# $$ P_2(x) = y_0 L_0(x) + y_1 L_1(x) + y_2 L_2(x) $$
# onde
# $$ L_0(x) = \frac{(x-x_1)(x-x_2)}{(x_0-x_1)(x_0-x_2)} $$
# $$ L_1(x) = \frac{(x-x_0)(x-x_2)}{(x_1-x_0)(x_1-x_2)} $$
# $$ L_2(x) = \frac{(x-x_0)(x-x_1)}{(x_2-x_0)(x_2-x_1)} $$

# %%


def lagrange_p2(x_to_interpolate, x_known_points, y_known_points):
    """
    Calcula a interpolação de Lagrange de 2º grau.
    x_to_interpolate: O ponto x onde a interpolação é desejada.
    x_known_points: Uma lista ou tupla com os 3 valores conhecidos de x (x0, x1, x2).
    y_known_points: Uma lista ou tupla com os 3 valores conhecidos de y (y0, y1, y2).
    Retorna o valor interpolado P₂(x_to_interpolate).
    """
    if len(x_known_points) != 3 or len(y_known_points) != 3:
        raise ValueError(
            "Para interpolação de Lagrange de 2º grau, são necessários exatamente 3 pontos (x, y).")

    x0, x1, x2 = x_known_points[0], x_known_points[1], x_known_points[2]
    y0, y1, y2 = y_known_points[0], y_known_points[1], y_known_points[2]

    # Calculando os termos de Lagrange L0(x), L1(x), L2(x)
    l0_x = ((x_to_interpolate - x1) * (x_to_interpolate - x2)) / \
        ((x0 - x1) * (x0 - x2))
    l1_x = ((x_to_interpolate - x0) * (x_to_interpolate - x2)) / \
        ((x1 - x0) * (x1 - x2))
    l2_x = ((x_to_interpolate - x0) * (x_to_interpolate - x1)) / \
        ((x2 - x0) * (x2 - x1))

    # Calculando o valor interpolado P₂(x)
    p2_x_interpolated = y0 * l0_x + y1 * l1_x + y2 * l2_x
    return p2_x_interpolated

# %% [markdown]
# ### Cálculo de $P_2(0.25)$


# %%
# Ponto onde queremos interpolar
x_interpolate_at = 0.25
p2_025_interpolated_value = None  # Inicializar

if y_coords_interp is not None:  # Verifica se y_coords_interp foi definido corretamente
    p2_025_interpolated_value = lagrange_p2(
        x_interpolate_at, x_coords_interp, y_coords_interp)
    print(f"Os pontos x usados para interpolação foram: {x_coords_interp}")
    print(
        f"Os pontos f(x) correspondentes (calculados com e^(3x) da tabela) foram: [{', '.join(f'{val:.5f}' for val in y_coords_interp)}]")
    print(
        f"O valor interpolado P₂({x_interpolate_at}) é: {p2_025_interpolated_value:.7f}")
else:
    print("Não foi possível calcular P₂(0.25) pois os pontos de interpolação (y_coords_interp) não foram definidos corretamente.")

# %% [markdown]
# ## Resultados Finais (Entrega)

# %%
print("Entrega:")
# Usando 7 casas decimais para consistência
print(f"i) f(0.25): {f_025_calculated_direct:.7f}")

if p2_025_interpolated_value is not None:
    # Usando 7 casas decimais
    print(f"ii) P₂(0.25): {p2_025_interpolated_value:.7f}")
else:
    print(f"ii) P₂(0.25): (Cálculo não pôde ser concluído devido a erro na seleção de pontos)")

print("-" * 40)
print("Nota: A escolha dos 3 pontos para a interpolação de Lagrange (P₂) foi baseada na")
print("suposição de que os 'valores sinalizados de negrito' seriam os mais próximos de 0.25,")
print(
    f"resultando na escolha de x = {x_coords_interp if 'x_coords_interp' in locals() else '[NÃO DEFINIDO]'}.")
print("Se a identificação dos valores em negrito for diferente, a lista 'x_coords_interp'")
print("na célula correspondente do notebook deve ser atualizada.")
