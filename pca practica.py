import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos de vinos
data = load_wine()
X = data.data

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Graficar resultado
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data.target, cmap='viridis')
plt.title("Resultados PCA para el conjunto de datos de Vinos")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.colorbar(label="Clases")
plt.show()