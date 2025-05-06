# %% [markdown]
# # Questão 2: Limite Superior para o Erro de Interpolação
#
# Esta questão pede para calcular o limite superior para o erro quando avaliamos $f(0.25)$ usando o polinômio interpolador $P_2(x)$ da Questão 1.
# A função da Questão 1 é $f(x) = x e^{3x}$.
# Para $P_2(x)$, foram usados 3 pontos ($n=2$).
#
# **Entrega:**
# * Derivada $f^{(n+1)}(x)$: ______________ "Escrever a equação usada"
# * ERRO (0.25): ______________ "Precisão de 4 casas decimais"
#
# A fórmula do erro é $E_n(x) = (x-x_0)(x-x_1)...(x-x_n) \frac{f^{(n+1)}(\xi)}{(n+1)!}$.
# Para $n=2$, temos $E_2(x) = (x-x_0)(x-x_1)(x-x_2) \frac{f'''(\xi)}{3!}$.

# %% [markdown]
# ## Importações e Definições Iniciais

# %%
import math
import sympy

# Configurações do Sympy para melhor visualização
sympy.init_printing(use_latex='mathjax')

# %% [markdown]
# ### Definição da Função e Pontos de Interpolação (da Questão 1)
#
# * Função: $f(x) = x e^{3x}$
# * Polinômio usado na Questão 1 para $f(0.25)$ foi $P_2(x)$, então $n=2$.
# * Ponto de avaliação: $x = 0.25$.
# * Pontos de interpolação assumidos para $P_2(x)$ (conforme notebook da Questão 1):
#   * $x_0 = 0.2$
#   * $x_1 = 0.3$
#   * $x_2 = 0.4$

# %%
# Definindo a variável simbólica e a função f(x) com sympy
x_sym = sympy.Symbol('x')
f_sym = x_sym * sympy.exp(3*x_sym)

# Pontos de interpolação e ponto de avaliação
x_eval = 0.25
x_nodes = [0.2, 0.3, 0.4]  # x0, x1, x2

# Ordem do polinômio n
n_order = 2

# %% [markdown]
# ## 1. Cálculo da Derivada $f^{(n+1)}(x)$
#
# Como $n=2$, precisamos calcular a $(n+1)$-ésima derivada, ou seja, $f'''(x)$.

# %%
# Calculando as derivadas simbolicamente
f_prime_sym = sympy.diff(f_sym, x_sym)
f_double_prime_sym = sympy.diff(f_prime_sym, x_sym)
f_triple_prime_sym = sympy.diff(f_double_prime_sym, x_sym)

# Simplificando as expressões das derivadas
f_prime_simplified = sympy.simplify(f_prime_sym)
f_double_prime_simplified = sympy.simplify(f_double_prime_sym)
f_triple_prime_simplified = sympy.simplify(f_triple_prime_sym)

print("f(x):")
display(f_sym)
print("f'(x):")
display(f_prime_simplified)
print("f''(x):")
display(f_double_prime_simplified)
print("f'''(x) (que é f^(n+1)(x) para n=2):")
display(f_triple_prime_simplified)

# Equação da derivada f'''(x) para a entrega
equacao_derivada_f_n_plus_1 = f_triple_prime_simplified

# %% [markdown]
# A equação para $f^{(n+1)}(x) = f'''(x)$ é $27(x+1)e^{3x}$.

# %% [markdown]
# ## 2. Cálculo do Limite Superior para o ERRO(0.25)
#
# O limite superior do erro é dado por:
# $|E_n(x)| \le \frac{\max_{\xi \in I} |f^{(n+1)}(\xi)|}{(n+1)!} \left| \prod_{i=0}^{n} (x-x_i) \right|$
#
# Para $n=2$:
# $|E_2(0.25)| \le \frac{\max_{\xi \in I} |f'''(\xi)|}{3!} |(0.25-x_0)(0.25-x_1)(0.25-x_2)|$
#
# O intervalo $I$ para $\xi$ deve conter $x_0, x_1, x_2$ e $x_{eval}$.
# Com $x_0=0.2, x_1=0.3, x_2=0.4$ e $x_{eval}=0.25$, o intervalo $I = [0.2, 0.4]$.
#
# ### Encontrar $\max_{\xi \in [0.2, 0.4]} |f'''(\xi)|$
#
# $f'''(x) = 27(x+1)e^{3x}$.
# Para encontrar o máximo, analisamos a próxima derivada, $f^{(4)}(x)$.

# %%
f_fourth_prime_sym = sympy.diff(f_triple_prime_sym, x_sym)
f_fourth_prime_simplified = sympy.simplify(f_fourth_prime_sym)

print("f^(4)(x):")
display(f_fourth_prime_simplified)

# Converter f'''(x) e f^(4)(x) para funções numéricas para avaliação
f_triple_prime_func = sympy.lambdify(x_sym, f_triple_prime_simplified, 'math')
f_fourth_prime_func = sympy.lambdify(x_sym, f_fourth_prime_simplified, 'math')

# Analisar o sinal de f^(4)(x) no intervalo [0.2, 0.4]
# Se f^(4)(x) > 0, então f'''(x) é crescente.
# Se f^(4)(x) < 0, então f'''(x) é decrescente.
# Valores de f^(4)(x) nos extremos do intervalo:
val_f4_at_02 = f_fourth_prime_func(0.2)
val_f4_at_04 = f_fourth_prime_func(0.4)

print(f"\nf^(4)(0.2) = {val_f4_at_02}")
print(f"f^(4)(0.4) = {val_f4_at_04}")

# Como f^(4)(x) = 27e^(3x)(3x+4), e para x em [0.2, 0.4], e^(3x)>0 e (3x+4)>0,
# então f^(4)(x) > 0 em todo o intervalo [0.2, 0.4].
# Isso significa que f'''(x) é estritamente crescente em [0.2, 0.4].

# Além disso, f'''(x) = 27(x+1)e^(3x) é positiva para x em [0.2, 0.4].
# Portanto, o máximo de |f'''(x)| ocorre em x = 0.4.
M_n_plus_1 = f_triple_prime_func(0.4)
print(
    f"\nMáximo de |f'''(ξ)| no intervalo [0.2, 0.4] é M₃ = f'''(0.4) = {M_n_plus_1}")

# %% [markdown]
# ### Cálculo do Termo Produtório e do Fatorial

# %%
# Termo produtório: (x-x0)(x-x1)(x-x2)
term_prod = (x_eval - x_nodes[0]) * (x_eval -
                                     x_nodes[1]) * (x_eval - x_nodes[2])
print(f"Termo produtório Π(x-x_i) = {term_prod}")

# Fatorial (n+1)! = 3!
n_plus_1_factorial = math.factorial(n_order + 1)
print(f"(n+1)! = {n_order+1}! = {n_plus_1_factorial}")

# %% [markdown]
# ### Cálculo do Limite Superior do Erro

# %%
erro_limite_superior = (M_n_plus_1 / n_plus_1_factorial) * abs(term_prod)
print(f"Limite superior para o erro |E₂({x_eval})| ≤ {erro_limite_superior}")

# Precisão de 4 casas decimais para o resultado final do erro
erro_final_4_casas = round(erro_limite_superior, 4)  # Arredondamento padrão
# Para garantir que seja um limite superior, podemos truncar para cima (ceil) na 4a casa,
# mas a questão pede "Precisão de 4 casas decimais" que usualmente implica arredondamento.
# Vamos usar o arredondamento padrão. Se for necessário garantir o limite superior estrito,
# ajustaríamos para cima. Ex: 0.007843... arredonda para 0.0078.

print(
    f"\nERRO ({x_eval}) com precisão de 4 casas decimais: {erro_final_4_casas}")

# %% [markdown]
# ## Entrega da Questão 2

# %%
# Reformatando a equação da derivada para string
str_equacao_derivada = str(equacao_derivada_f_n_plus_1).replace(
    '**', '^').replace('*', '')

print("Entrega:")
print(f"Derivada f'''(x): {str_equacao_derivada}     (Equação: 27(x+1)e^(3x))")
print(f"ERRO ({x_eval}): {erro_final_4_casas}                     (Precisão de 4 casas decimais)")

# %% [markdown]
# ---
# **Nota sobre a precisão:** O valor calculado para o limite superior do erro foi `0.00784315...`. Arredondado para 4 casas decimais, torna-se `0.0078`.
# Se a intenção fosse um valor que *garantidamente* é maior ou igual ao erro, e o resultado exato fosse, por exemplo, $0.00781$, o arredondamento para $0.0078$ ainda seria um limite superior. Se fosse $0.00788$, arredondaria para $0.0079$. O arredondamento padrão geralmente é aceitável para "precisão de X casas".
