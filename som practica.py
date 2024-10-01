from sklearn.datasets import load_digits
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de dígitos
digits = load_digits()
X = digits.data

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicializar y entrenar SOM
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 100)

# Mostrar el mapa de hits
plt.figure(figsize=(7, 7))
plt.pcolor(som.distance_map().T, cmap='Blues')  # Mapa de distancias
plt.colorbar()
plt.title("Mapa Autoorganizado (SOM) para el conjunto de datos de Dígitos")

# Añadir marcas para los winning nodes
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Encontrar el nodo ganador
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', 
             markeredgecolor='r', markersize=12, markeredgewidth=2)
plt.show()