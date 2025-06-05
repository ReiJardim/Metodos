
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr

# Dados experimentais
strain = np.array([0.15, 0.16, 0.18, 0.19, 0.22, 0.27, 0.31, 0.36, 0.41, 0.5,
                   0.61, 0.72, 0.83, 0.92, 1.02, 1.13, 1.24, 1.35, 1.46, 1.51])
stress = np.array([0.92, 1.00, 1.12, 1.17, 1.28, 1.42, 1.53, 1.67, 1.78, 1.92,
                   2.10, 2.28, 2.42, 2.51, 2.56, 2.51, 2.42, 2.28, 2.10, 1.85])

# Separação por faixa
x_a, y_a = strain[0:10], stress[0:10]
x_b, y_b = strain[10:17], stress[10:17]
x_c, y_c = strain[17:20], stress[17:20]

# Regressão Linear (a)
model_a = LinearRegression().fit(x_a.reshape(-1, 1), y_a)
r_a, _ = pearsonr(x_a, y_a)
print("Regressão Linear (a): y = {:.4f}x + {:.4f}".format(model_a.coef_[0], model_a.intercept_))
print("Correlação de Pearson (a): r = {:.4f}".format(r_a))

# Regressão Quadrática (b)
poly = PolynomialFeatures(degree=2)
X_b_poly = poly.fit_transform(x_b.reshape(-1, 1))
model_b = LinearRegression().fit(X_b_poly, y_b)
y_b_pred = model_b.predict(X_b_poly)
r_b, _ = pearsonr(y_b, y_b_pred)
print("Regressão Quadrática (b): y = {:.4f}x² + {:.4f}x + {:.4f}".format(model_b.coef_[2], model_b.coef_[1], model_b.intercept_))
print("Correlação de Pearson (b): r = {:.4f}".format(r_b))

# Regressão Linear (c)
model_c = LinearRegression().fit(x_c.reshape(-1, 1), y_c)
r_c, _ = pearsonr(x_c, y_c)
print("Regressão Linear (c): y = {:.4f}x + {:.4f}".format(model_c.coef_[0], model_c.intercept_))
print("Correlação de Pearson (c): r = {:.4f}".format(r_c))

# Gerando gráficos
x_plot_a = np.linspace(min(x_a), max(x_a), 100)
y_plot_a = model_a.predict(x_plot_a.reshape(-1, 1))

x_plot_b = np.linspace(min(x_b), max(x_b), 100)
X_plot_b_poly = poly.transform(x_plot_b.reshape(-1, 1))
y_plot_b = model_b.predict(X_plot_b_poly)

x_plot_c = np.linspace(min(x_c), max(x_c), 100)
y_plot_c = model_c.predict(x_plot_c.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(strain, stress, color='black', label='Dados experimentais')
plt.plot(x_plot_a, y_plot_a, color='blue', label='Regressão Linear (a)')
plt.plot(x_plot_b, y_plot_b, color='green', label='Regressão Quadrática (b)')
plt.plot(x_plot_c, y_plot_c, color='red', label='Regressão Linear (c)')
plt.xlabel('Deformação (ε)')
plt.ylabel('Tensão (σ)')
plt.title('Ajuste de Curvas: Tração/Deformação')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
